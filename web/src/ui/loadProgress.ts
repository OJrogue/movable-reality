// What the loading card shows, from the bytes Spark's workers report per file. Sizes have to be
// known up front: only four workers load at a time, so a total summed from the files already
// started would grow as the others begin, and the bar would run backwards. The hosted proxy also
// compresses full responses and drops their Content-Length, so sizes come from a one-byte Range
// request (probeSize).

/** Share of the bar for the download; the rest is decoding and the first LOD cut. */
const BYTES_SHARE = 0.95;

interface FileState {
  size: number | null;
  loaded: number;
  done: boolean;
}

export interface LoadSnapshot {
  /** 0 to 1, never falls. Stays under 1 until the card calls complete(). */
  fraction: number;
  loadedBytes: number;
  /** Null until every file's size is known. */
  totalBytes: number | null;
  filesDone: number;
  files: number;
}

export class LoadProgress {
  private readonly files = new Map<string, FileState>();
  private shown = 0;

  /** Register a file. Registering it again keeps what is known about it. */
  add(url: string): void {
    if (!this.files.has(url)) this.files.set(url, { size: null, loaded: 0, done: false });
  }

  setSize(url: string, size: number): void {
    const f = this.files.get(url);
    if (f && size > 0) f.size = size;
  }

  /** From Spark's onProgress. `total` is 0 when the response had no Content-Length. */
  update(url: string, loaded: number, total: number): void {
    const f = this.files.get(url);
    if (!f) return;
    f.loaded = Math.max(f.loaded, loaded);
    if (total > 0) f.size = total;
  }

  /** The file has finished, loaded or failed; either way the bar stops waiting for it. */
  settle(url: string): void {
    const f = this.files.get(url);
    if (f) f.done = true;
  }

  allSettled(): boolean {
    return this.files.size > 0 && [...this.files.values()].every((f) => f.done);
  }

  complete(): void {
    this.shown = 1;
  }

  snapshot(): LoadSnapshot {
    const all = [...this.files.values()];
    const known = all.filter((f) => f.size !== null);
    // A file of unknown size counts as an average one until its size arrives.
    const average = known.length ? known.reduce((s, f) => s + f.size!, 0) / known.length : 1;
    let weight = 0;
    let got = 0;
    let loadedBytes = 0;
    for (const f of all) {
      const size = f.size ?? average;
      weight += size;
      got += f.done ? size : f.size === null ? 0 : Math.min(f.loaded, size);
      loadedBytes += f.done && f.size !== null ? f.size : f.loaded;
    }
    if (weight > 0) this.shown = Math.max(this.shown, (BYTES_SHARE * got) / weight);
    const unknown = all.length === 0 || known.length < all.length;
    return {
      fraction: this.shown,
      loadedBytes,
      totalBytes: unknown ? null : known.reduce((s, f) => s + f.size!, 0),
      filesDone: all.filter((f) => f.done).length,
      files: all.length,
    };
  }
}

/** The line under the room's title. */
export function loadingText(s: LoadSnapshot): string {
  if (s.files === 0) return "Loading the room";
  if (s.filesDone === s.files) return "Setting up the room";
  const percent = `${Math.floor(s.fraction * 100)}%`;
  if (s.totalBytes === null) return `${percent} · ${s.filesDone} of ${s.files} files`;
  const mb = (n: number) => Math.round(n / 1e6);
  return `${percent} · ${Math.min(mb(s.loadedBytes), mb(s.totalBytes))} of ${mb(s.totalBytes)} MB`;
}

/** Total size from a Content-Range header such as "bytes 0-0/8751088". */
export function sizeFromContentRange(header: string | null): number | null {
  const m = /\/(\d+)\s*$/.exec(header ?? "");
  return m ? Number(m[1]) : null;
}

/** A file's size from a one-byte Range request. Null if the server does not answer with one. */
export async function probeSize(url: string): Promise<number | null> {
  const r = await fetch(url, { headers: { Range: "bytes=0-0" } });
  r.body?.cancel().catch(() => undefined);
  return r.status === 206 ? sizeFromContentRange(r.headers.get("content-range")) : null;
}
