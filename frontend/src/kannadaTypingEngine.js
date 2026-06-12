// src/kannadaTypingEngine.js
const HALANT = "್";

// Consonant base map (greedy: longer keys first when matching)
const CONSONANT_MAP = {
  "ksh": "ಕ್ಷ",
  "kh": "ಖ", "k": "ಕ",
  "gh": "ಘ", "g": "ಗ",
  "ng": "ಙ",
  "ch": "ಚ", "c": "ಚ", "chh": "ಛ",
  "jh": "ಝ", "j": "ಜ",
  "ny": "ಞ",
  "th": "ಥ", "t": "ತ",
  "dh": "ಧ", "d": "ದ",
  "ph": "ಫ", "p": "ಪ",
  "bh": "ಭ", "b": "ಬ",
  "m": "ಮ",
  "y": "ಯ",
  "r": "ರ",
  "l": "ಲ",
  "v": "ವ", "w": "ವ",
  "sh": "ಶ", "ss": "ಷ", "s": "ಸ",
  "h": "ಹ",
  "ll": "ಳ", "L": "ಳ",
  "q": "ಕ್", "z": "ಝ", "x": "ಕ್ಸ"
};

// Vowel matra when attached to a consonant (empty for 'a' because inherent vowel)
const VOWEL_MATRA = {
  "aa": "ಾ",
  "a": "",
  "i": "ಿ",
  "ii": "ೀ",
  "ee": "ೀ",
  "u": "ು",
  "uu": "ೂ",
  "o": "ೊ",
  "oo": "ೋ",
  "e": "ೆ",
  "ai": "ೈ",
  "au": "ೌ",
  "ru": "ೃ"
};

// Standalone vowels
const VOWEL_STANDALONE = {
  "aa": "ಆ",
  "a": "ಅ",
  "i": "ಇ",
  "ii": "ಈ",
  "u": "ಉ",
  "uu": "ಊ",
  "e": "ಎ",
  "ee": "ಏ",
  "ai": "ಐ",
  "o": "ಒ",
  "oo": "ಓ",
  "au": "ಔ",
  "ru": "ಋ"
};

const consonantKeys = Object.keys(CONSONANT_MAP).sort((a,b) => b.length - a.length);
const vowelKeys = Object.keys(VOWEL_MATRA).sort((a,b) => b.length - a.length);
const vowelStandaloneKeys = Object.keys(VOWEL_STANDALONE).sort((a,b) => b.length - a.length);

function normalizeRoman(s) {
  return s.normalize("NFKC").toLowerCase();
}

function transliterateToken(token) {
  if (!token) return "";
  token = normalizeRoman(token);

  let i = 0;
  let out = "";

  while (i < token.length) {
    // match vowel standalone first
    let vm = null;
    for (const vk of vowelStandaloneKeys) {
      if (token.startsWith(vk, i)) { vm = vk; break; }
    }
    if (vm) {
      out += VOWEL_STANDALONE[vm] || "";
      i += vm.length;
      continue;
    }

    // match consonant
    let cm = null;
    for (const ck of consonantKeys) {
      if (token.startsWith(ck, i)) { cm = ck; break; }
    }

    if (!cm) {
      // fallback - pass unknown char as-is (but keep ascii)
      i += 1;
      continue;
    }

    const consKannada = CONSONANT_MAP[cm] || "";
    i += cm.length;

    // try vowel after consonant
    let va = null;
    for (const vk of vowelKeys) {
      if (token.startsWith(vk, i)) { va = vk; break; }
    }

    if (va) {
      const matra = VOWEL_MATRA[va] ?? "";
      if (matra === "") {
        out += consKannada; // inherent vowel
      } else {
        out += consKannada + matra;
      }
      i += va.length;
    } else {
      // dead consonant -> add halant
      out += consKannada + HALANT;
    }
  }

  // remove trailing halant if present (conservative)
  if (out.endsWith(HALANT)) out = out.slice(0, -1);
  return out;
}

// transliterate trailing ASCII token of fullInput
export function phoneticKannada(fullInput) {
  if (!fullInput) return "";
  const m = fullInput.match(/([a-zA-Z]+)$/);
  if (!m) return fullInput;
  const token = m[1];
  const prefix = fullInput.slice(0, -token.length);
  const translit = transliterateToken(token);
  return prefix + translit;
}

// apply matra: if text ends with a matra char and previous is consonant, ensure correct join
export const applyMatra = (text) => {
  if (!text || text.length < 2) return text;
  const last = text.slice(-1);
  const prev = text.slice(-2, -1);
  if (Object.values(VOWEL_MATRA).includes(last) && /[ಕ-ಹ]/.test(prev)) {
    return text.slice(0, -2) + prev + last;
  }
  return text;
};

// auto halant join heuristic
export const autoHalant = (text) => {
  if (!text || text.length < 2) return text;
  const last = text.slice(-1);
  const prev = text.slice(-2, -1);
  if (/[ಕ-ಹ]/.test(prev) && /[ಕ-ಹ]/.test(last)) {
    return text.slice(0, -1) + HALANT + last;
  }
  return text;
};

export default phoneticKannada;
