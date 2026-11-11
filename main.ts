/**
 * make.fun photo-geolocator-01 entry agent.
 *
 * One model call per photo (POST to {OPENAI_BASE_URL}/chat/completions).
 * Every photo gets an answer: if the model cannot place it, a fixed fallback
 * of latitude 50, longitude 10 (central Europe) is used.
 * Transient errors (408, 429, 5xx) are retried up to 3 times with a fixed
 * 2000ms*(attempt+1) backoff; gateway refusals (any other 4xx) are final and
 * fall back to (50, 10).
 * Deterministic: no Date.now, no Math.random, no wall clock anywhere.
 */

type Photo = { id: string | number; url: string; answered: boolean };
type Coord = { lat: number | null; lon: number | null };

const MAKEFUN_ENV_URL = process.env.MAKEFUN_ENV_URL;
const OPENAI_BASE_URL = process.env.OPENAI_BASE_URL;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

if (!MAKEFUN_ENV_URL || !OPENAI_BASE_URL || !OPENAI_API_KEY) {
  throw new Error(
    'Missing required env vars: MAKEFUN_ENV_URL, OPENAI_BASE_URL, OPENAI_API_KEY',
  );
}

const envUrl = MAKEFUN_ENV_URL.replace(/\/+$/, '');
const openaiBaseUrl = OPENAI_BASE_URL.replace(/\/+$/, '');

const PROMPT =
  'You are a geolocation expert. Look carefully at this street-level photo and determine where on Earth it was taken. First, in 2-4 short sentences, note the strongest visual clues, in this order: (1) any text on signs, shop names, license plates or road markings — what language and alphabet; (2) architecture and building styles; (3) road layout, which side traffic drives on, road markings and traffic signs; (4) vegetation, climate and terrain; (5) vehicles, people, clothing and any flags or distinctive objects. Then, on its own final line, give your single best estimate as JSON: {"lat": <degrees north>, "lon": <degrees east>}. If the clues are weak, still give your best guess — never refuse and never leave the coordinates out.';

const TRANSIENT_STATUSES = new Set([408, 429, 500, 502, 503, 504, 529]);

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function base64(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function fetchJson<T>(url: string, init: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} for ${url}`);
  }
  return (await res.json()) as T;
}

async function fetchPhotos(): Promise<Photo[]> {
  const data = await fetchJson<{ photos: Photo[] }>(
    `${envUrl}/photos`,
    { method: 'GET' },
  );
  return data.photos;
}

async function fetchPhotoBytes(url: string): Promise<Uint8Array | null> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${envUrl}${url}`);
      if (res.ok) {
        const ab = await res.arrayBuffer();
        return new Uint8Array(ab);
      }
      if (TRANSIENT_STATUSES.has(res.status)) {
        lastError = new Error(`HTTP ${res.status} fetching photo ${url}`);
        await delay(1000 * (attempt + 1));
        continue;
      }
      return null;
    } catch (err) {
      lastError = err;
      await delay(1000 * (attempt + 1));
    }
  }
  return null;
}

async function askModel(jpegBase64: string): Promise<string | null> {
  const body = {
    model: 'gpt-6.1-sol',
    max_tokens: 1024,
    messages: [
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${jpegBase64}` } },
          { type: 'text', text: PROMPT },
        ],
      },
    ],
  };

  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${openaiBaseUrl}/chat/completions`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${OPENAI_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      const shouldRetryHeader = res.headers.get('x-should-retry');
      const forceNoRetry = shouldRetryHeader === 'false';

      if (!res.ok) {
        if (TRANSIENT_STATUSES.has(res.status) && !forceNoRetry) {
          lastError = new Error(`HTTP ${res.status} from model`);
          await delay(2000 * (attempt + 1));
          continue;
        }
        // Gateway refusal or any other 4xx — final.
        const text = await res.text().catch(() => '');
        console.error(`Model request failed: HTTP ${res.status} ${text}`);
        return null;
      }

      const data = await res.json();
      const choice = data.choices?.[0];
      if (!choice) return null;
      const content = choice.message?.content;
      if (content === undefined || content === null) return null;
      if (typeof content === 'string') return content;
      if (Array.isArray(content)) {
        return content
          .filter((part) => part.type === 'text')
          .map((part) => (part as { text?: string }).text ?? '')
          .join('');
      }
      return String(content);
    } catch (err) {
      lastError = err;
      await delay(2000 * (attempt + 1));
    }
  }
  return null;
}

function extractJsonObjects(text: string): { lat: unknown; lon: unknown }[] {
  const objects: { lat: unknown; lon: unknown }[] = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  let escaped = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === '\\') {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === '{') {
      if (depth === 0) start = i;
      depth++;
    } else if (ch === '}') {
      if (depth > 0) {
        depth--;
        if (depth === 0 && start >= 0) {
          try {
            const obj = JSON.parse(text.slice(start, i + 1));
            if (obj && typeof obj === 'object') {
              objects.push(obj as { lat: unknown; lon: unknown });
            }
          } catch {
            // ignore malformed fragment
          }
          start = -1;
        }
      }
    }
  }
  return objects;
}

function parseLatLonFromValue(v: unknown): number | null {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string') {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function parseCoordFromObject(obj: { lat: unknown; lon: unknown }): Coord | null {
  const latKeys = ['lat', 'latitude'];
  const lonKeys = ['lon', 'lng', 'longitude'];
  let lat: number | null = null;
  let lon: number | null = null;
  for (const k of latKeys) {
    if (k in obj) { lat = parseLatLonFromValue(obj[k]); break; }
  }
  for (const k of lonKeys) {
    if (k in obj) { lon = parseLatLonFromValue(obj[k]); break; }
  }
  if (lat === null || lon === null) return null;
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) return null;
  return { lat, lon };
}

function findFirstConsecutivePair(text: string): Coord | null {
  const nums: number[] = [];
  let i = 0;
  const len = text.length;
  while (i < len) {
    const ch = text[i];
    if (ch === '-' || (ch >= '0' && ch <= '9') || ch === '.') {
      let start = i;
      if (ch === '-') {
        i++;
        if (i >= len || !(text[i] >= '0' && text[i] <= '9')) continue;
      }
      let dotSeen = false;
      while (i < len && ((text[i] >= '0' && text[i] <= '9') || (text[i] === '.' && !dotSeen))) {
        if (text[i] === '.') dotSeen = true;
        i++;
      }
      const n = Number(text.slice(start, i));
      if (Number.isFinite(n)) nums.push(n);
    } else {
      i++;
    }
  }
  for (let j = 0; j + 1 < nums.length; j++) {
    const a = nums[j];
    const b = nums[j + 1];
    if (a >= -90 && a <= 90 && b >= -180 && b <= 180) {
      return { lat: a, lon: b };
    }
  }
  return null;
}

function parseCoordinates(text: string): Coord | null {
  const objects = extractJsonObjects(text);
  for (let i = objects.length - 1; i >= 0; i--) {
    const parsed = parseCoordFromObject(objects[i]);
    if (parsed) return parsed;
  }
  return findFirstConsecutivePair(text);
}

async function postAnswers(answers: { id: string | number; lat: number; lon: number }[]): Promise<boolean> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(`${envUrl}/answers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers }),
      });
      if (res.ok) {
        console.log(`Answers POST succeeded: HTTP ${res.status}`);
        return true;
      }
      if (TRANSIENT_STATUSES.has(res.status)) {
        lastError = new Error(`HTTP ${res.status} posting answers`);
        await delay(2000 * (attempt + 1));
        continue;
      }
      const text = await res.text().catch(() => '');
      console.error(`Answers POST failed: HTTP ${res.status} ${text}`);
      return false;
    } catch (err) {
      lastError = err;
      await delay(2000 * (attempt + 1));
    }
  }
  console.error('Answers POST failed after retries:', lastError);
  return false;
}

async function processPhoto(photo: Photo): Promise<{ id: string | number; lat: number; lon: number; byModel: boolean }> {
  const fallback = { id: photo.id, lat: 50, lon: 10, byModel: false };

  const bytes = await fetchPhotoBytes(photo.url);
  if (!bytes) return fallback;

  const jpegBase64 = base64(bytes);
  const reply = await askModel(jpegBase64);
  if (!reply) return fallback;

  const coord = parseCoordinates(reply);
  if (!coord) return fallback;

  return { id: photo.id, lat: coord.lat, lon: coord.lon, byModel: true };
}

async function main(): Promise<void> {
  const photos = await fetchPhotos();
  const unanswered = photos.filter((p) => !p.answered);

  const concurrency = 4;
  const results: Array<{ id: string | number; lat: number; lon: number; byModel: boolean }> = [];

  for (let i = 0; i < unanswered.length; i += concurrency) {
    const batch = unanswered.slice(i, i + concurrency);
    const batchResults = await Promise.all(batch.map(processPhoto));
    results.push(...batchResults);
  }

  const byModel = results.filter((r) => r.byModel).length;
  const byFallback = results.length - byModel;
  console.log(`Placed ${byModel} photos by model, ${byFallback} by fallback.`);

  const answers = results.map((r) => ({
    id: r.id,
    lat: Math.round(r.lat * 1e6) / 1e6,
    lon: Math.round(r.lon * 1e6) / 1e6,
  }));

  const ok = await postAnswers(answers);
  if (ok) {
    process.exit(0);
  } else {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err.stack ?? String(err));
  process.exit(1);
});
