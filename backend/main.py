import pathlib
import uuid
import boto3
from fastapi import Depends, HTTPException,status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import modal
import os
from pydantic import BaseModel

class ProcessVideoRequest(BaseModel):
     s3_key:str

image=(modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04",add_python="3.12")
       .apt_install(["ffmpeg","libgl1-mesa-glx","wget","libcudnn8","libcudnn8-dev"])
       .pip_install_from_requirements("requirements.txt")
       .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                      "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com.google/fonts/row/main/ofl/anton/Anton-Regular.ttf",
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
        pass

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        pass

    @modal.fastapi_endpoint(method="POST")
    def process_video(self,request:ProcessVideoRequest,token:HTTPAuthorizationCredentials=Depends(auth_scheme)):
        s3_key=request.s3_key

        if token.credentials!=os.environ["AUTH_TOKEN"]:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail="Incorret bearer token",headers={"WWW-Authenticate":"Bearer"})
        
        run_id= str(uuid.uuid4())
        base_dir=pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True,exist_ok=True)

        #Download video file
        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        s3_client.download_file("ai-podcast-clipper-tool-bucket", s3_key, str(video_path))


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