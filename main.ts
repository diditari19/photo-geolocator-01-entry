// photo-geolocator-01 entry: for each of the instance's 25 photos, ask
// gpt-6.1-sol once where it was taken, with a reasoning prompt that
// reads the visual evidence first, then parse the latitude and longitude.
// A photo whose model call is refused or yields no coordinates still gets
// an answer: a fixed prior (50, 10), central Europe, the densest
// Mapillary coverage region — a ~7,000 km error on average, far better
// than the 20,015 km an unanswered photo counts. Every photo is answered.
//
// Runs as the grader runs an entry, `node /bot/main.ts` on Node 22
// (erasable TypeScript, the standard library only), with the trial's
// variables: MAKEFUN_ENV_URL (the env), OPENAI_BASE_URL and
// OPENAI_API_KEY (the gateway and the dummy key it swaps for the real
// one). Model calls run on the OpenAI key pushed to the competition
// before submitting; with no key pushed every call is answered 401 and
// every photo falls back to the prior, still scored.
//
// Deterministic given the model's answers: every request is a function of
// the photo's bytes and the fixed prompt, the calls run four at a time and
// their answers are posted in photo order in a single POST, so a replay
// sends the same calls and leaves the same answers. No wall clock, no
// unseeded randomness anywhere in a request.

const MODEL = 'gpt-6.1-sol';
const MAX_TOKENS = 1024;
const CONCURRENCY = 4;
/** Fallback guess when a photo cannot be placed: central Europe (50, 10). */
const FALLBACK = { lat: 50, lon: 10 };
const PROMPT =
  'You are a geolocation expert. Look carefully at this street-level photo and determine where on Earth it was taken.\n' +
  'First, in 2-4 short sentences, note the strongest visual clues, in this order: (1) any text on signs, shop names, license plates or road markings — what language and alphabet; (2) architecture and building styles; (3) road layout, which side traffic drives on, road markings and traffic signs; (4) vegetation, climate and terrain; (5) vehicles, people, clothing and any flags or distinctive objects.\n' +
  'Then, on its own final line, give your single best estimate as JSON: {"lat": <degrees north>, "lon": <degrees east>}.\n' +
  'If the clues are weak, still give your best guess — never refuse and never leave the coordinates out.';
/** Retried with a fixed backoff: a provider's transient errors. A gateway refusal (4xx) is final. */
const RETRY_STATUSES = [408, 429, 500, 502, 503, 504, 529];
const RETRIES = 3;

interface Photo {
  id: string;
  url: string;
  answered: boolean;
}

interface Guess {
  id: string;
  lat: number;
  lon: number;
}

function need(name: string): string {
  const v = process.env[name];
  if (v === undefined || v === '') throw new Error(`${name} is not set`);
  return v;
}

const envUrl = need('MAKEFUN_ENV_URL').replace(/\/+$/, '');
const openaiUrl = need('OPENAI_BASE_URL').replace(/\/+$/, '');
const apiKey = need('OPENAI_API_KEY');

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** The model's answer to one photo, as text; null when the gateway refused the call (final). */
async function ask(jpeg: Buffer): Promise<string | null> {
  const body = JSON.stringify({
    model: MODEL,
    max_tokens: MAX_TOKENS,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${jpeg.toString('base64')}` } },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  });
  for (let attempt = 0; ; attempt += 1) {
    let res: Response;
    try {
      res = await fetch(`${openaiUrl}/chat/completions`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
        body,
      });
    } catch (err) {
      if (attempt < RETRIES) {
        await sleep(2000 * (attempt + 1));
        continue;
      }
      process.stderr.write(`geolocator: model call failed to send: ${(err as Error).message}\n`);
      return null;
    }
    if (res.ok) {
      const msg = (await res.json()) as { choices?: { message?: { content?: string | { type: string; text?: string }[] } }[] };
      const content = msg.choices?.[0]?.message?.content ?? '';
      return typeof content === 'string' ? content : content.filter((b) => b.type === 'text').map((b) => b.text ?? '').join('\n');
    }
    const text = await res.text();
    if (RETRY_STATUSES.includes(res.status) && res.headers.get('x-should-retry') !== 'false' && attempt < RETRIES) {
      await sleep(2000 * (attempt + 1));
      continue;
    }
    process.stderr.write(`geolocator: the model call was refused (HTTP ${res.status}): ${text.slice(0, 300)}\n`);
    return null;
  }
}

const isLat = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -90 && v <= 90;
const isLon = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && v >= -180 && v <= 180;

const asNumber = (v: unknown): number | null => {
  if (typeof v === 'number') return v;
  if (typeof v === 'string') {
    const n = Number(v.trim());
    return Number.isFinite(n) ? n : null;
  }
  return null;
};

/** The latitude and longitude an answer gives: its last JSON object with lat and lon, else its first two numbers in range. */
function coordinates(text: string): { lat: number; lon: number } | null {
  let found: { lat: number; lon: number } | null = null;
  for (const m of text.matchAll(/\{[^{}]*\}/g)) {
    try {
      const o = JSON.parse(m[0]) as Record<string, unknown>;
      const lat = asNumber(o.lat ?? o.latitude);
      const lon = asNumber(o.lon ?? o.lng ?? o.longitude);
      if (lat !== null && lon !== null && isLat(lat) && isLon(lon)) found = { lat, lon };
    } catch {
      // Not JSON: the next one, or the numbers below.
    }
  }
  if (found !== null) return found;
  const nums = [...text.matchAll(/-?\d+(?:\.\d+)?/g)].map((m) => Number(m[0]));
  for (let i = 0; i + 1 < nums.length; i += 1) {
    if (isLat(nums[i]) && isLon(nums[i + 1])) return { lat: nums[i]!, lon: nums[i + 1]! };
  }
  return null;
}

/** A photo's bytes; null when the env would not serve it after retries. */
async function fetchPhoto(url: string): Promise<Buffer | null> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      const res = await fetch(url);
      if (res.ok) return Buffer.from(await res.arrayBuffer());
      process.stderr.write(`geolocator: photo fetch HTTP ${res.status} for ${url}\n`);
    } catch (err) {
      process.stderr.write(`geolocator: photo fetch failed for ${url}: ${(err as Error).message}\n`);
    }
    if (attempt >= RETRIES) return null;
    await sleep(1000 * (attempt + 1));
  }
}

const round6 = (v: number): number => Math.round(v * 1e6) / 1e6;

async function postAnswers(answers: Guess[]): Promise<boolean> {
  const body = JSON.stringify({ answers });
  for (let attempt = 0; ; attempt += 1) {
    try {
      const res = await fetch(`${envUrl}/answers`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body,
      });
      const text = await res.text();
      process.stdout.write(`geolocator: POST /answers attempt ${attempt + 1}: HTTP ${res.status} ${text}\n`);
      if (res.ok) return true;
      if (!RETRY_STATUSES.includes(res.status) || attempt >= RETRIES) return false;
    } catch (err) {
      process.stderr.write(`geolocator: POST /answers failed to send: ${(err as Error).message}\n`);
      if (attempt >= RETRIES) return false;
    }
    await sleep(1000 * (attempt + 1));
  }
}

async function main(): Promise<number> {
  const listed = (await (await fetch(`${envUrl}/photos`)).json()) as { photos: Photo[] };
  const photos = listed.photos.filter((p) => !p.answered);
  const guesses: Guess[] = new Array(photos.length);
  let viaModel = 0;
  let viaFallback = 0;
  let next = 0;
  const worker = async (): Promise<void> => {
    while (next < photos.length) {
      const i = next;
      next += 1;
      const p = photos[i]!;
      let c: { lat: number; lon: number } | null = null;
      const jpeg = await fetchPhoto(`${envUrl}${p.url}`);
      if (jpeg !== null) {
        const text = await ask(jpeg);
        if (text !== null) c = coordinates(text);
        if (c === null) process.stderr.write(`geolocator: no coordinates for ${p.id}, falling back\n`);
      } else {
        process.stderr.write(`geolocator: could not fetch ${p.id}, falling back\n`);
      }
      if (c !== null) {
        viaModel += 1;
        guesses[i] = { id: p.id, lat: round6(c.lat), lon: round6(c.lon) };
      } else {
        viaFallback += 1;
        guesses[i] = { id: p.id, lat: FALLBACK.lat, lon: FALLBACK.lon };
      }
    }
  };
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));
  process.stdout.write(`geolocator: ${photos.length} photos: ${viaModel} placed by the model, ${viaFallback} fallback\n`);
  const ok = await postAnswers(guesses);
  return ok ? 0 : 1;
}

try {
  process.exitCode = await main();
} catch (err) {
  process.stderr.write(`geolocator: ${(err as Error).stack ?? String(err)}\n`);
  process.exitCode = 1;
}
