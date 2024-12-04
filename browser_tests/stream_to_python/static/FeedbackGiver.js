const serverhost = location.host;

export class FeedbackGiver {
  constructor(target, target_by_word, on_transcription, on_word_spoken) {
    this.target = target;
    this.target_by_word = target_by_word;
    this.transcription = "";
    this.on_transcription = on_transcription;
    this.socket = null;
    this.audioContext = null;
    this.audioWorkletNode = null;

    this.on_word_spoken = on_word_spoken;
    this.words = [];
    this.are_words_correct = [];
    for (const [word, _] of target_by_word) {
      this.words.push(word);
      this.are_words_correct.push(false);
    }
    this.next_word_ix = 0;
    this.recognition = null;
  }

  #setTranscription(transcription) {
    this.transcription = transcription;
    this.on_transcription(this.transcription);
  }

  async getFeedback() {
    const res = await fetch(
      `http://${serverhost}/feedback?target=${encodeURIComponent(
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
      `http://${serverhost}/score_words_cer?target=${encodeURIComponent(
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
      `http://${serverhost}/score_words_wfed?target=${encodeURIComponent(
        this.target
      )}&tbw=${encodeURIComponent(
        JSON.stringify(this.target_by_word)
      )}&speech=${encodeURIComponent(this.transcription)}`
    );
    const data = await res.json();
    const [scoredWords, overall] = data;
    return [scoredWords, overall];
  }

  async getSideBySideDescription() {
    const res = await fetch(
      `http://${serverhost}/side_by_side_description?target=${encodeURIComponent(`
        ${this.target}
        `)}&tbw=${encodeURIComponent(
        JSON.stringify(this.target_by_word)
      )}&speech=${encodeURIComponent(this.transcription)}`
    );
    return await res.json();
  }

  async start() {
    // Clear previous transcription
    this.#setTranscription("");

    // Open WebSocket connection
    this.socket = new WebSocket(`ws://${serverhost}/stream`);

    // Handle incoming transcriptions
    this.socket.onmessage = async (event) => {
      this.#setTranscription(event.data);
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

    this.#startWordTranscription();
  }

  async stop() {
    if (this.audioWorkletNode) {
      this.audioWorkletNode.disconnect();
    }
    if (this.socket) {
      this.socket.close();
    }
    if (this.recognition) {
      this.recognition.onend = null;
      this.recognition.stop();
    }
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
  }

  #startWordTranscription() {
    this.next_word_ix = 0;
    for (let i = 0; i < this.words.length; i++) {
      this.are_words_correct[i] = false;
    }

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    this.recognition = new SpeechRecognition();
    this.recognition.lang = "en-US";
    this.recognition.interimResults = true;
    this.recognition.maxAlternatives = 1;
    this.recognition.start();
    this.recognition.onend = () => this.recognition.start();
    const finalWords = [];
    let numCorrect = 0;
    this.recognition.onresult = (event) => {
      const wordlist = [...event.results]
        .map((result) => result[0].transcript.split(" "))
        .reduce((a, b) => a.concat(b))
        .filter((w) => w.length > 0);
      const isFinal = [...event.results].at(-1).isFinal;
      if (isFinal) {
        finalWords.push(...wordlist);
        wordlist.length = 0;
      }

      const allWords = finalWords.concat(wordlist);
      if (
        allWords[0].toLowerCase().replace(/[^a-z]/g, "") !=
          this.words[0].toLowerCase().replace(/[^a-z]/g, "") &&
        allWords.length > 0
      ) {
        allWords.shift();
      }
      numCorrect = 0;
      for (let i = 0; i < allWords.length; i++) {
        const word = allWords[i];
        const target = this.words[i];

        if (
          word.toLowerCase().replace(/[^a-z]/g, "") ===
          target.toLowerCase().replace(/[^a-z]/g, "")
        ) {
          this.are_words_correct[i] = true;
          numCorrect++;
        } else {
          this.are_words_correct[i] = false;
        }
        this.next_word_ix = i + 1;
        if (this.next_word_ix < this.words.length) {
          this.on_word_spoken(
            this.words,
            this.are_words_correct,
            this.next_word_ix,
            Math.round((1000 * numCorrect) / this.next_word_ix) / 10,
            false
          );
        } else {
          this.recognition.onend = null;
          this.recognition.stop();
          this.on_word_spoken(
            this.words,
            this.are_words_correct,
            this.next_word_ix,
            Math.round((1000 * numCorrect) / this.words.length) / 10,
            true
          );
        }
      }
    };
  }
}
