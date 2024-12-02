export class FeedbackGiver {
  constructor(target, target_by_word, on_transcription) {
    this.target = target;
    this.target_by_word = target_by_word;
    this.transcription = "";
    this.on_transcription = on_transcription;
    this.socket = null;
    this.audioContext = null;
    this.audioWorkletNode = null;
  }

  set_transcription(transcription) {
    this.transcription = transcription;
    this.on_transcription(this.transcription);
  }

  async getFeedback() {
    const res = await fetch(
      `http://${location.host}/feedback?target=${encodeURIComponent(
        this.target
      )}&tbw=${encodeURIComponent(
        JSON.stringify(this.target_by_word)
      )}&speech=${encodeURIComponent(this.transcription)}`
    );
    const data = await res.json();
    const [perWordFeedback, top3feedback] = data;
    return [perWordFeedback, top3feedback];
  }

  async getCER() {
    const res = await fetch(
      `http://${location.host}/score_words_cer?target=${encodeURIComponent(
        this.target
      )}&tbw=${encodeURIComponent(
        JSON.stringify(this.target_by_word)
      )}&speech=${encodeURIComponent(this.transcription)}`
    );
    const data = await res.json();
    const [scoredWords, overall] = data;
    return [scoredWords, overall];
  }

  async getWFED() {
    const res = await fetch(
      `http://${location.host}/score_words_wfed?target=${encodeURIComponent(
        this.target
      )}&tbw=${encodeURIComponent(
        JSON.stringify(this.target_by_word)
      )}&speech=${encodeURIComponent(this.transcription)}`
    );
    const data = await res.json();
    const [scoredWords, overall] = data;
    return [scoredWords, overall];
  }

  async start() {
    // Clear previous transcription
    this.set_transcription("");

    // Open WebSocket connection
    this.socket = new WebSocket(`ws://${location.host}/stream`);

    // Handle incoming transcriptions
    this.socket.onmessage = async (event) => {
      this.set_transcription(event.data);
    };

    // Start capturing audio
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: true,
    });

    // Create an AudioContext
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 16000,
      latencyHint: "interactive",
    });

    // Load the AudioWorkletProcessor (which handles audio processing)
    await this.audioContext.audioWorklet.addModule("WavWorklet.js");

    // Create the AudioWorkletNode
    this.audioWorkletNode = new AudioWorkletNode(
      this.audioContext,
      "wav-worklet"
    );

    // Connect the audio input to the AudioWorkletNode
    const audioInput = this.audioContext.createMediaStreamSource(stream);
    audioInput.connect(this.audioWorkletNode);

    // Connect the AudioWorkletNode to the audio context destination
    this.audioWorkletNode.connect(this.audioContext.destination);

    // Connect AudioWorkletNode to process audio and send to WebSocket
    this.audioWorkletNode.port.onmessage = (event) => {
      if (this.socket.readyState === WebSocket.OPEN) {
        this.socket.send(event.data);
      }
    };
  }

  async stop() {
    if (this.audioWorkletNode) {
      this.audioWorkletNode.disconnect();
    }
    if (this.socket) {
      this.socket.close();
    }
    if (this.audioContext) {
      await this.audioContext.close();
    }
  }
}
