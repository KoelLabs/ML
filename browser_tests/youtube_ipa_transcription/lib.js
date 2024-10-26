// taken from https://github.com/darenliang/darenliang.com/blob/master/misc/ytdl-ffmpeg-demo/main.js
// but converted to not run in the browser to enable caching the youtube video data (since it is super slow to download :/)

const ytdl = require("ytdl-core");
const fs = require("fs/promises");
const util = require("util");
const exec = util.promisify(require("child_process").exec);

fs.mkdir("./cache").catch(() => {});
async function getCached(filename) {
  filename = filename.replace(/[^a-zA-Z0-9\.]/g, "");
  if (
    await fs
      .access(`./cache/${filename}`)
      .then(() => true)
      .catch(() => false)
  ) {
    return await fs.readFile(`./cache/${filename}`);
  } else {
    return null;
  }
}
async function setCached(filename, data) {
  filename = filename.replace(/[^a-zA-Z0-9\.]/g, "");
  await fs.writeFile(`./cache/${filename}`, data);
  // if more than 100 files, delete the 10 oldest ones
  const files = await fs.readdir("./cache");
  if (files.length > 100) {
    const stats = await Promise.all(
      files.map(async (file) => ({
        file,
        stat: await fs.stat(`./cache/${file}`),
      }))
    );
    stats.sort((a, b) => a.stat.mtimeMs - b.stat.mtimeMs);
    for (let i = 0; i < 10; i++) {
      await fs.unlink(`./cache/${stats[i].file}`);
    }
  }
}

function proxy(url) {
  return url;
  // return "https://[PROXY URL GOES HERE]/?" + encodeURIComponent(url);
}

async function getInfo(url) {
  // cannot cache since youtube urls expire
  const re =
    /(https?:\/\/)?(((m|www)\.)?(youtube(-nocookie)?|youtube.googleapis)\.com.*(v\/|v=|vi=|vi\/|e\/|embed\/|shorts\/|user\/.*\/u\/\d+\/)|youtu\.be\/)([_0-9a-z-]+)/i;
  const videoId = url.match(re)[8];

  const headers = {
    "X-YouTube-Client-Name": "5",
    "X-YouTube-Client-Version": "19.09.3",
    "User-Agent":
      "com.google.ios.youtube/19.09.3 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)",
    "Content-Type": "application/json",
  };

  const b = {
    context: {
      client: {
        clientName: "IOS",
        clientVersion: "19.09.3",
        deviceModel: "iPhone14,3",
        userAgent:
          "com.google.ios.youtube/19.09.3 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)",
        hl: "en",
        timeZone: "UTC",
        utcOffsetMinutes: 0,
      },
    },
    videoId,
    playbackContext: {
      contentPlaybackContext: { html5Preference: "HTML5_PREF_WANTS" },
    },
    contentCheckOk: true,
    racyCheckOk: true,
  };

  console.log("[info] getting info");
  const data = await fetch(
    proxy(
      `https://www.youtube.com/youtubei/v1/player?key=${process.env.YOUTUBE_API_KEY}&prettyPrint=false`
    ),
    {
      method: "POST",
      body: JSON.stringify(b),
      headers,
    }
  ).then((r) => r.json());

  // patch formats
  data.streamingData.adaptiveFormats = augmentFormats(
    data.streamingData.adaptiveFormats
  );

  return data;
}

async function getImportantInfo(url) {
  const cached = await getCached(`${url}.json`);
  if (cached) {
    return JSON.parse(cached);
  }

  const info = await getInfo(url);
  const result = {
    formats: info.streamingData.adaptiveFormats,
    videoDetails: info.videoDetails,
    endscreen: info.endscreen,
    captions: info.captions,
  };

  setCached(`${url}.json`, JSON.stringify(result));
  return result;
}

function augmentFormats(formats) {
  const fn = (format) => {
    const [type, codecsPart] = format.mimeType.split(";");
    const container = type.split("/")[1].trim(); // 'mp4'
    const codecs = codecsPart.match(/codecs="(.+)"/)[1]; // 'mp4a.40.2'
    const hasVideo =
      codecs.includes("avc1") ||
      codecs.includes("vp8") ||
      codecs.includes("vp9");
    const hasAudio =
      codecs.includes("mp4a") ||
      codecs.includes("opus") ||
      codecs.includes("vorbis");
    return {
      ...format,
      container,
      hasVideo,
      hasAudio,
      codecs,
      videoCodec: hasVideo ? codecs : undefined,
      audioCodec: hasAudio ? codecs : undefined,
      isLive: false,
      isHLS: false,
      isDashMPD: false,
    };
  };

  return formats.map((format) => fn(format));
}

async function getAudioBuffer(audioInfo) {
  console.log("[info] fetching data");
  const audioData = await fetch(proxy(audioInfo.url)).then((r) =>
    r.arrayBuffer().then((r) => Buffer.from(r))
  );

  console.log("[info] writing data");
  const audioFilename = `audio.${audioInfo.container}`;
  await fs.writeFile(audioFilename, audioData);

  console.log("[info] encoding as mp3");
  await exec(`ffmpeg -i ${audioFilename} output.mp3`);

  console.log("[info] unlink temporary file");
  await fs.unlink(audioFilename);

  console.log("[info] sending final data");
  const result = await fs.readFile("output.mp3");
  await fs.unlink("output.mp3");
  return result;
}

async function getVideoBuffer(videoInfo, audioInfo = null) {
  console.log("[info] fetching data");
  let videoData, audioData, audioFilename;
  if (audioInfo) {
    [videoData, audioData] = await Promise.all([
      fetch(proxy(videoInfo.url)).then((r) =>
        r.arrayBuffer().then((r) => Buffer.from(r))
      ),
      fetch(proxy(audioInfo.url)).then((r) =>
        r.arrayBuffer().then((r) => Buffer.from(r))
      ),
    ]);
    audioFilename = `audio.${audioInfo.container}`;
    await fs.writeFile(audioFilename, audioData);
  } else {
    videoData = await fetch(proxy(videoInfo.url)).then((r) =>
      r.arrayBuffer().then((r) => Buffer.from(r))
    );
  }

  console.log("[info] writing data");
  const videoFilename = `video.${videoInfo.container}`;
  await fs.writeFile(videoFilename, videoData);

  console.log("[info] encoding as mp4");
  if (audioInfo) {
    await exec(
      `ffmpeg -i ${videoFilename} -i ${audioFilename} -c:v copy -c:a copy -shortest output.mp4`
    );
    await fs.unlink(audioFilename);
  } else {
    await exec(`ffmpeg -i ${videoFilename} output.mp4`);
  }

  console.log("[info] unlink temporary files");
  await fs.unlink(videoFilename);

  console.log("[info] sending final data");
  const result = await fs.readFile("output.mp4");
  await fs.unlink("output.mp4");
  return result;
}

async function getFastVideoBuffer(url) {
  const cached = await getCached(`${url}.mp4`);
  if (cached) {
    return cached;
  }

  const info = await getInfo(url);

  console.log("[info] choosing formats");
  const formats = info.streamingData.adaptiveFormats;
  const videoInfo = ytdl.chooseFormat(formats, {
    quality: "highest",
    filter: (format) => format.container === "mp4",
  });

  console.log("[info] fetching data");
  const videoData = await fetch(proxy(videoInfo.url)).then((r) =>
    r.arrayBuffer()
  );

  console.log("[info] sending data");
  const result = Buffer.from(videoData);
  setCached(`${url}.mp4`, result);
  return result;
}

async function getBestVideoBuffer(url) {
  const cached = await getCached(`${url}.best.mp4`);
  if (cached) {
    return cached;
  }

  const info = await getInfo(url);

  console.log("[info] choosing formats");
  const formats = info.streamingData.adaptiveFormats;
  const videoInfo = ytdl.chooseFormat(formats, {
    quality: "highestvideo",
  });
  const audioInfo = ytdl.chooseFormat(formats, {
    quality: "highestaudio",
    filter: "audioonly",
  });

  const result = await getVideoBuffer(videoInfo, audioInfo);
  setCached(`${url}.best.mp4`, result);
  return result;
}

async function getBestAudioBuffer(url) {
  const cached = await getCached(`${url}.best.mp3`);
  if (cached) {
    return cached;
  }

  const info = await getInfo(url);

  console.log("[info] choosing format");
  const formats = info.streamingData.adaptiveFormats;
  const audioInfo = ytdl.chooseFormat(formats, {
    quality: "highestaudio",
    filter: "audioonly",
  });

  const result = await getAudioBuffer(audioInfo);
  setCached(`${url}.best.mp3`, result);
  return result;
}

module.exports = {
  getImportantInfo,
  getFastVideoBuffer,
  getBestVideoBuffer,
  getBestAudioBuffer,
};
