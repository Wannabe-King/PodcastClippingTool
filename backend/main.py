import json
import pathlib
import pickle
import shutil
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
from google import genai

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


app=modal.App("ai-podcast-clipper",image=image)

volume= modal.Volume.from_name(
    "ai-podcast-clipper-model-cache",create_if_missing=True
)

mount_path="/root/.cache/torch"

auth_scheme= HTTPBearer()

def process_clip(base_dir: str, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):
    clip_name=f"clip_{clip_index}"
    s3_key_dir= os.path.dirname(s3_key)
    output_s3_key= f"{s3_key_dir}/{clip_name}.mp4"
    print("Output s3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    #Segment path: original clip from start to end time
    clip_segment_path= clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path= clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_ouput_path= clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time
    # command to cut video into the clip from start_time to end_time
    cut_command = (f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)
    
    # command to extract audio from the clipped segment
    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True,
                   check=True, capture_output=True)
    

    # copy the clip to base directory for processing from colombia asd model
    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

    # asd columbia command for active speaker detection -> outputs tracks and scores in pickel format
    # pickel is just a json object to store python object
    columbia_command = (f"python Columbia_test.py --videoName {clip_name} "
                        f"--videoFolder {str(base_dir)} "
                        f"--pretrainModel weight/finetuning_TalkSet.model")
    
    columbia_start_time = time.time()
    subprocess.run(columbia_command, cwd="/asd", shell=True)
    columbia_end_time = time.time()
    print(
        f"Columbia script completed in {columbia_end_time - columbia_start_time:.2f} seconds")
    
    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or scores not found for clip")

    # open both tracks and scores file and load them into memory
    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    # cvv_start_time = time.time()
    # create_vertical_video(
    #     tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path
    # )
    # cvv_end_time = time.time()
    # print(
    #     f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds")

    # create_subtitles_with_ffmpeg(transcript_segments, start_time,
    #                              end_time, vertical_mp4_path, subtitle_output_path, max_words=5)

    # s3_client = boto3.client("s3")
    # s3_client.upload_file(
    #     subtitle_output_path, "ai-podcast-clipper", output_s3_key)



@app.cls(gpu="L40S",timeout=900,retries=0,scaledown_window=20,secrets=[modal.Secret.from_name("ai-podcast-clipper-secret")],volumes={mount_path:volume})
class AiPodcastClipper:
    @modal.enter()
    def load_model(self):
        # Loading whisperX model
        self.whisperx_model=whisperx.load_model("large-v2",device="cuda",compute_type="float16")

        self.alignment_model,self.metadata=whisperx.load_align_model(language_code="en",device="cuda")
        print("Transcription models loaded ...")

        print("Creating gemini client...")
        self.gemini_client= genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        print("Created gemini client...")

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path= base_dir / "audio.wav"
        extract_cmd= f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd,shell=True,check=True,capture_output=True)

        print("Starting transcription with WhisperX...")
        start_time= time.time()

        audio= whisperx.load_audio(audio_path)
        result= self.whisperx_model.transcribe(audio, batch_size=16)

        # Align: synchronizing the recognized text segments with the audio at a more precise level
        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

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

        return json.dumps(segments)

    def donwload_video(self,s3_key: str,base_dir:str) -> str:
        
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

    def identify_moments(self,transcript: dict):
        response = self.gemini_client.models.generate_content(model="gemini-2.5-flash-preview", contents="""
    This is a podcast video transcript consisting of word, along with each words's start and end time. I am looking to create clips between a minimum of 30 and maximum of 60 seconds long. The clip should never exceed 60 seconds.

    Your task is to find and extract stories, or question and their corresponding answers from the transcript.
    Each clip should begin with the question and conclude with the answer.
    It is acceptable for the clip to include a few additional sentences before a question if it aids in contextualizing the question.

    Please adhere to the following rules:
    - Ensure that clips do not overlap with one another.
    - Start and end timestamps of the clips should align perfectly with the sentence boundaries in the transcript.
    - Only use the start and end timestamps provided in the input. modifying timestamps is not allowed.
    - Format the output as a list of JSON objects, each representing a clip with 'start' and 'end' timestamps: [{"start": seconds, "end": seconds}, ...clip2, clip3]. The output should always be readable by the python json.loads function.
    - Aim to generate longer clips between 40-60 seconds, and ensure to include as much content from the context as viable.

    Avoid including:
    - Moments of greeting, thanking, or saying goodbye.
    - Non-question and answer interactions.

    If there are no valid clips to extract, the output should be an empty list [], in JSON format. Also readable by json.loads() in Python.

    The transcript is as follows:\n\n""" + str(transcript))
        print(f"Identified moments response: ${response.text}")
        return response.text


    @modal.fastapi_endpoint(method="POST")
    def process_video(self,request:ProcessVideoRequest,token:HTTPAuthorizationCredentials=Depends(auth_scheme)):
        s3_key=request.s3_key
        
        #Step 1: Authenticate Request
        if token.credentials!=os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Incorret bearer token",headers={"WWW-Authenticate":"Bearer"})

        run_id= str(uuid.uuid4())
        base_dir=pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True,exist_ok=True)

        # #Step 2: Download video file
        # video_path= self.donwload_video(s3_key,base_dir)
        video_path=base_dir / "input.mp4"
        s3_client= boto3.client("s3")
        s3_client.download_file("ai-podcast-clipper-tool-bucket",s3_key,str(video_path))

        # #Step 3: Transcription
        transcript_segment_json = self.transcribe_video(base_dir,video_path)
        transcript_segment= json.loads(transcript_segment_json)
        
        # #Step 4: Indentify moments for clips
        print("Identifying clip moments")
        identified_moments_raw = self.identify_moments(transcript_segment)
        cleaned_json_string = identified_moments_raw.strip()
        if cleaned_json_string.startswith("```json"):
            cleaned_json_string = cleaned_json_string[len("```json"):].strip()
        if cleaned_json_string.endswith("```"):
            cleaned_json_string = cleaned_json_string[:-len("```")].strip()

        clip_moments= json.loads(cleaned_json_string)
        if not clip_moments or not isinstance(clip_moments, list):
            print("Error: Identified moments is not a list")
            clip_moments = []

        print(clip_moments)

        # #Step 5: Process Clips
        for index, moment in enumerate(clip_moments[:3]):
            if "start" in moment and "end" in moment:
                print("Processing clip" + str(index) + " from " + str(moment["start"]) + " to "+ str(moment["end"]))
                process_clip(base_dir, video_path, s3_key, moment["start"], moment["end"], index, transcript_segment)

        if base_dir.exists():
            print("Cleaning up tem dir after "+base_dir)
            shutil.rmtree(base_dir,ignore_errors=True)



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