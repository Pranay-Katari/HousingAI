const fs = require("fs");
const path = require("path");
const { S3Client, GetObjectCommand, PutObjectCommand } = require("@aws-sdk/client-s3");
const { BedrockRuntimeClient, InvokeModelCommand } = require("@aws-sdk/client-bedrock-runtime");
const csvParser = require("csv-parser");

const REGION = "us-east-1";

const s3 = new S3Client({ region: REGION });
const bedrock = new BedrockRuntimeClient({ region: REGION });

const outputBucket = process.env.OUTPUT_BUCKET || "local-output-bucket";

function streamCsv(stream) {
  return new Promise((resolve, reject) => {
    const rows = [];
    stream
      .pipe(csvParser())
      .on("data", (data) => rows.push(data))
      .on("end", () => resolve(rows))
      .on("error", reject);
  });
}

async function invokeBedrockModel(row) {
  const promptText = `<REDACTED>`;

  const payload = {
    inferenceConfig: {
      max_new_tokens: 1000
    },
    messages: [
      {
        role: "user",
        content: [
          {
            text: promptText
          }
        ]
      }
    ]
  };

  try {
    const response = await bedrock.send(new InvokeModelCommand({
      modelId: "amazon.nova-micro-v1:0",     
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify(payload),
    }));

    const body = await response.body.transformToString();
    const parsed = JSON.parse(body);
    const reply = parsed.output?.message?.content?.[0]?.text?.trim();
    const extracted = reply?.match(/\*\*(.*?)\*\*/)?.[1];
    return extracted;
  } catch (err) {
    return "Error invoking model";
  }
}

async function superviseWithTitan(row, agentAnswer) {
  const prompt = '<REDACTED>`;

  const payload = {
    inferenceConfig: {
      max_new_tokens: 20
    },
    messages: [
      {
        role: "user",
        content: [{ text: prompt }]
      }
    ]
  };

  try {
    const response = await bedrock.send(new InvokeModelCommand({
      modelId: "amazon.nova-micro-v1:0",
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify(payload),
    }));

    const body = await response.body.transformToString();
    const parsed = JSON.parse(body);
    const reply = parsed.output?.message?.content?.[0]?.text?.trim().toLowerCase();
    return reply === "yes";
  } catch (err) {
    return false;
  }
}

async function getValidatedTax(row) {
  const maxAttempts = 3;
  let attempt = 0;
  let answer = null;
  let approved = false;

  while (attempt < maxAttempts && !approved) {
    attempt++;
    answer = await invokeBedrockModel(row);

    if (!answer) {
      continue;
    }

    approved = await superviseWithTitan(row, answer);

    if (!approved) {
    }
  }

  return answer;
}

exports.handler = async (event) => {
  let inputStream;
  let outputKey;

  if (event.Records?.[0]) {
    const record = event.Records[0];
    const bucket = record.s3.bucket.name;
    const key = decodeURIComponent(record.s3.object.key.replace(/\+/g, " "));
    const object = await s3.send(new GetObjectCommand({ Bucket: bucket, Key: key }));
    inputStream = object.Body;
    outputKey = `results/${Date.now()}_results.csv`;
  } else if (event.localFilePath) {
    inputStream = fs.createReadStream(path.resolve(event.localFilePath));
    outputKey = `output/${Date.now()}_results.csv`;
  } else {
    throw new Error("No valid input source found.");
  }

  const rows = await streamCsv(inputStream);
  const results = [];

  for (const row of rows) {
    const t = await getValidatedTax(row);
    results.push({ ...row, t});
  }

  const header = "<REDACTED>";
  const csvBody = results.map(r =>
    `${r.<REDACTED>},${r.<REDACTED>},${r.<REDACTED>},${r.<REDACTED>}`
  ).join("\n");
  const finalCsv = header + csvBody;

  if (event.localFilePath) {
    const outputPath = path.resolve(outputKey);
    fs.mkdirSync(path.dirname(outputPath), { recursive: true });
    fs.writeFileSync(outputPath, finalCsv);
  } else {
    await s3.send(new PutObjectCommand({
      Bucket: outputBucket,
      Key: outputKey,
      Body: finalCsv,
      ContentType: "text/csv",
    }));
  }
};
