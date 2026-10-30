// Lane ui: microphone capture for hold-to-talk. An AudioWorklet copies each 128-sample
// render quantum to the main thread. The worklet code is inlined as a Blob URL, so the
// build needs no separate asset.
import { chunksToWav } from "./audio";

const WORKLET = `
class MrTap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("mr-tap", MrTap);
`;

interface Session {
  ctx: AudioContext;
  stream: MediaStream;
  node: AudioWorkletNode;
  chunks: Float32Array[];
}

export class Recorder {
  private starting: Promise<Session> | null = null;

  get recording(): boolean {
    return this.starting !== null;
  }

  /** Asks for the microphone (once per page) and starts collecting samples. */
  start(): Promise<void> {
    if (!this.starting) {
      const p = this.open();
      this.starting = p;
      p.catch(() => {
        if (this.starting === p) this.starting = null; // denied or no microphone
      });
    }
    return this.starting.then(() => undefined);
  }

  private async open(): Promise<Session> {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    const ctx = new AudioContext();
    await ctx.resume(); // created after an await, so Safari may start it suspended
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
    try {
      await ctx.audioWorklet.addModule(url);
    } finally {
      URL.revokeObjectURL(url);
    }
    const node = new AudioWorkletNode(ctx, "mr-tap", { numberOfInputs: 1, numberOfOutputs: 0 });
    const chunks: Float32Array[] = [];
    node.port.onmessage = (e: MessageEvent<Float32Array>) => chunks.push(e.data);
    ctx.createMediaStreamSource(stream).connect(node);
    return { ctx, stream, node, chunks };
  }

  /** Stops recording and returns a 16 kHz mono 16-bit WAV plus its duration in seconds. */
  async stop(): Promise<{ wav: ArrayBuffer; seconds: number }> {
    const starting = this.starting;
    this.starting = null;
    if (!starting) throw new Error("not recording");
    const s = await starting;
    s.node.port.onmessage = null;
    s.node.disconnect();
    for (const t of s.stream.getTracks()) t.stop(); // releases the microphone indicator
    const rate = s.ctx.sampleRate;
    await s.ctx.close();
    const samples = s.chunks.reduce((n, c) => n + c.length, 0);
    return { wav: chunksToWav(s.chunks, rate), seconds: samples / rate };
  }
}
