class WavWorklet extends AudioWorkletProcessor {
  constructor() {
    super();
    this.port.onmessage = this.handleMessage.bind(this);
  }

  process(inputs, outputs, parameters) {
    // convert to single channel but keep float32, post to main thread
    const input = inputs[0]; // Input is a 2D array: channels x samples
    if (input.length > 0) {
      const numChannels = input.length;
      const numSamples = input[0].length;

      // Create an array to hold mixed single channel data
      const mixedData = new Float32Array(numSamples);

      // Mix all channels to a single channel by averaging (or summing if preferred)
      for (let i = 0; i < numSamples; i++) {
        let sum = 0;
        for (let j = 0; j < numChannels; j++) {
          sum += input[j][i]; // Add the sample from each channel
        }
        mixedData[i] = sum / numChannels; // Average the samples from all channels
      }

      // send to main thread
      this.port.postMessage(mixedData);
    }

    // Return true to keep the processor alive
    return true;
  }

  handleMessage(event) {
    // Handle incoming messages if necessary
  }
}

registerProcessor("wav-worklet", WavWorklet);
