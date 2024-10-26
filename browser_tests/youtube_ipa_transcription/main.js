const express = require("express");
const app = express();
const port = 8000;

const {
  getImportantInfo,
  getFastVideoBuffer,
  getBestVideoBuffer,
  getBestAudioBuffer,
} = require("./lib");

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/index.html");
});

app.get("/transformers.js", (req, res) => {
  res.sendFile(__dirname + "/transformers.js");
});

// serve models folder
app.use("/models", express.static(__dirname + "/models"));

app.get("/info", async (req, res) => {
  const url = req.query.url;
  const info = await getImportantInfo(url);
  res.json(info);
});

app.get("/fast-video", async (req, res) => {
  const url = req.query.url;
  const buffer = await getFastVideoBuffer(url);
  res.send(buffer);
});

app.get("/best-video", async (req, res) => {
  const url = req.query.url;
  const buffer = await getBestVideoBuffer(url);
  res.send(buffer);
});

app.get("/best-audio", async (req, res) => {
  const url = req.query.url;
  const buffer = await getBestAudioBuffer(url);
  res.send(buffer);
});

app.listen(port, () => {
  console.log(`Example app listening on port ${port}`);
});
