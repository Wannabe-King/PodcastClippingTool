import { env } from "~/env";
import { inngest } from "./client";
import { db } from "~/server/db";
import { ListObjectsV2Command, S3Client } from "@aws-sdk/client-s3";

export const processVideo = inngest.createFunction(
  {
    id: "process-video",
    retries: 1,
    concurrency: {
      limit: 1,
      key: "event.data.userId",
    },
  },
  { event: "process-video-events" },
  // every steps here are retriable piece of logics up until defined retires
  // it helps divide complex workflow to discrete runnable tasks independent of each other
  async ({ event, step }) => {
    const { uploadedFileId } = event.data;

    // Checking how many credits the user has
    const { userId, credits, s3Key } = await step.run(
      "check-credits",
      async () => {
        const uploadFile = await db.uploadedFile.findUniqueOrThrow({
          where: {
            id: uploadedFileId,
          },
          select: {
            user: {
              select: {
                id: true,
                credits: true,
              },
            },
            s3Key: true,
          },
        });

        return {
          userId: uploadFile.user.id,
          credits: uploadFile.user.credits,
          s3Key: uploadFile.s3Key,
        };
      },
    );

    if (credits > 0) {
      // Set the upload file status to processing
      await step.run("set-status-processing", async () => {
        await db.uploadedFile.update({
          where: {
            id: uploadedFileId,
          },
          data: {
            status: "processing",
          },
        });
      });

      // Calling the modal backend endpoint
      await step.run("call-modal-endpoint", async () => {
        await fetch(env.PROCESS_VIDEO_ENDPOINT, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${env.AUTH_SECRET}`,
          },
          body: JSON.stringify({ s3_key: s3Key }),
        });
      });

      // Update the db with data about the created clips by backend
      const { clipsFound } = await step.run("create-clips-in-db", async () => {
        const folderPrefix = s3Key.split("/")[0]!;

        // Get all the files present in the prefix folder
        const allKeys = await listS3ObjectsByPrefix(folderPrefix);

        // Find all the clips that are added
        const clipKeys = allKeys.filter(
          (key): key is string =>
            key !== undefined && !key.endsWith("original.mp4"),
        );

        if (clipKeys.length > 0) {
          await db.clip.createMany({
            data: clipKeys.map((clipKey) => ({
              s3Key: clipKey,
              uploadedFileId,
              userId,
            })),
          });
        }

        return { clipsFound: clipKeys.length };
      });

      await step.run("deduct-credits", async () => {
        await db.user.update({
          where: {
            id: userId,
          },
          data: {
            credits: {
              // We deduct the credit if present
              // if not enough just use all that is left
              decrement: Math.min(credits, clipsFound),
            },
          },
        });
      });

      // Set the upload file status to processed
      await step.run("set-status-processed", async () => {
        await db.uploadedFile.update({
          where: {
            id: uploadedFileId,
          },
          data: {
            status: "processed",
          },
        });
      });
    } else {
      // Set the upload file status to no-credits
      await step.run("set-status-no-credits", async () => {
        await db.uploadedFile.update({
          where: {
            id: uploadedFileId,
          },
          data: {
            status: "no credits",
          },
        });
      });
    }
  },
);

// Getting list of all the s3 object in the given path
async function listS3ObjectsByPrefix(prefix: string) {
  const s3Client = new S3Client({
    region: env.AWS_REGION,
    credentials: {
      accessKeyId: env.AWS_ACCESS_KEY_ID,
      secretAccessKey: env.AWS_SECRET_ACCESS_KEY,
    },
  });

  const listCommand = new ListObjectsV2Command({
    Bucket: env.S3_BUCKET_NAME,
    Prefix: prefix,
  });

  const response = await s3Client.send(listCommand);
  return response.Contents?.map((item) => item.Key).filter(Boolean) ?? [];
}
