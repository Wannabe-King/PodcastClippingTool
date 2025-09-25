import json
import pathlib
import subprocess
import time
import uuid
import boto3
import botocore
from fastapi import Depends, HTTPException,status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import modal
import os
from pydantic import BaseModel
import whisperx

class ProcessVideoRequest(BaseModel):
     s3_key:str

image=(modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04",add_python="3.12")
       .apt_install(["ffmpeg","libgl1-mesa-glx","wget","libcudnn8","libcudnn8-dev"])
       .pip_install_from_requirements("requirements.txt")
       .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                      "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                      "fc-cache -f -v"])
                      .add_local_dir("asd","/asd",copy=True))


app=modal.App("backend",image=image)

volume= modal.Volume.from_name(
    "ai-podcast-clipper-model-cache",create_if_missing=True
)

mount_path="/root/.cache/torch"

auth_scheme= HTTPBearer()

@app.cls(gpu="L40S",timeout=900,retries=0,scaledown_window=20,secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")],volumes={mount_path:volume})
class AiPodcastClipper:
    @modal.enter()
    def load_model(self):
        print("Loading models")
        self.whisperx_model=whisperx.load_model("large-v2",device="cuda",compute_type="float16")

        self.alignment_model,self.metadata=whisperx.load_align_model(language_code="en",device="cuda")
        print("Transcription models loaded ...")

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path= base_dir / "audio.wav"
        extract_cmd= f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd,shell=True,check=True,capture_output=True)

        print("Starting transcription with WhisperX...")
        start_time= time.time()

        audio= whisperx.load_audio(audio_path)
        result= self.whisperx_model.transcribe(audio, batch_size=16)

        result = whisperx.align(result["segments"],self.alignment_model,self.metadata,audio,device="cuda",return_char_alignments=False)

        duration= time.time() - start_time

        print("Transcription and alignment took "+ str(duration) + "seconds")
        segments=[]

        if "word_segments" in result:
            for word_segments in result["word_segments"]:
                segments.append({
                    "start":word_segments["start"],
                    "end":word_segments["end"],
                    "word":word_segments["word"],
                })

    def donwload_video(self,s3_key: str,base_dir:str) -> str:
        #Download video file
        try:
            video_path = base_dir / "input.mp4"
            s3_client = boto3.client("s3")
            s3_client.download_file("ai-podcast-clipper-tool-bucket", s3_key, str(video_path))
            return video_path
        except botocore.exceptions.ClientError as error:
            error_code = error.response['Error']['Code']
            error_message = error.response['Error']['Message']
        
            print(f"S3 Error {error_code}: {error_message}")
        
            if error_code == '404':
                raise HTTPException(status_code=404, detail=f"Video file not found: {s3_key}")
            elif error_code == '403':
                raise HTTPException(status_code=403, detail=f"Access denied to video file: {s3_key}")
            else:
                raise HTTPException(status_code=500, detail=f"Failed to download video: {error_message}")
    
        except Exception as error:
            print(f"Unexpected error downloading video: {error}")
            raise HTTPException(status_code=500, detail="Failed to download video from S3")


    @modal.fastapi_endpoint(method="POST")
    def process_video(self,request:ProcessVideoRequest,token:HTTPAuthorizationCredentials=Depends(auth_scheme)):
        s3_key=request.s3_key

        if token.credentials!=os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Incorret bearer token",headers={"WWW-Authenticate":"Bearer"})

        run_id= str(uuid.uuid4())
        base_dir=pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True,exist_ok=True)

        video_path= self.donwload_video(s3_key,base_dir)

        self.transcribe_video(base_dir,video_path)
        print(os.listdir(base_dir))
        

@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clipper=AiPodcastClipper()
    url= ai_podcast_clipper.process_video.web_url
    payload={
        "s3_key":"test1/mi65min.mp4"
    }
    headers= {
        "Content-Type":"application/json",
        "Authorization":"Bearer 123123"
    }

    response= requests.post(url,json=payload,headers=headers)
    result=response.json()
    print(result)