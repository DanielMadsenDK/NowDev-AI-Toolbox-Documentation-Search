// Query analysis for the lexical (FTS5) search leg: stopword removal, identifier splitting, and
// ServiceNow alias/acronym expansion. Pure functions so the database layer only has to plan and run queries.

const STOPWORDS = new Set([
  "a", "about", "above", "after", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been",
  "before", "being", "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing", "done", "during",
  "each", "either", "else", "etc", "even", "ever", "every", "few", "for", "from", "further", "had", "has", "have",
  "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
  "just", "let", "lets", "like", "may", "me", "might", "more", "most", "much", "must", "my", "need", "needs", "no",
  "nor", "of", "off", "on", "once", "one", "only", "or", "other", "our", "ours", "out", "over", "own", "please",
  "same", "servicenow", "shall", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
  "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "us",
  "very", "via", "want", "was", "way", "we", "were", "what", "when", "where", "whether", "which", "while", "who",
  "whom", "why", "will", "with", "within", "without", "would", "you", "your", "yours",
]);

// Shorthand globals and objects used in ServiceNow scripts, mapped to the API object they document.
const IDENTIFIER_ALIASES: Record<string, string[]> = {
  gs: ["GlideSystem"],
  g_form: ["GlideForm"],
  g_user: ["GlideUser"],
  g_list: ["GlideList"],
  g_scratchpad: ["scratchpad"],
  g_navigation: ["GlideNavigation"],
  g_modal: ["GlideModal"],
  gr: ["GlideRecord"],
  gdt: ["GlideDateTime"],
  ga: ["GlideAjax", "GlideAggregate"],
  sn_ws: ["RESTMessageV2", "SOAPMessageV2"],
};

// Acronyms and jargon expanded on the lexical leg only; the embedding model sees the user's own wording.
const ACRONYMS: Record<string, string[]> = {
  acl: ["access control"],
  acls: ["access control"],
  atf: ["automated test framework"],
  br: ["business rule"],
  brs: ["business rules"],
  si: ["script include"],
  cmdb: ["configuration management database"],
  ci: ["configuration item"],
  cis: ["configuration items"],
  sla: ["service level agreement"],
  slas: ["service level agreements"],
  ola: ["operational level agreement"],
  itsm: ["it service management"],
  itom: ["it operations management"],
  itam: ["it asset management"],
  itbm: ["it business management"],
  spm: ["strategic portfolio management"],
  ppm: ["project portfolio management"],
  sam: ["software asset management"],
  ham: ["hardware asset management"],
  csm: ["customer service management"],
  fsm: ["field service management"],
  hrsd: ["hr service delivery"],
  irm: ["integrated risk management"],
  grc: ["governance risk and compliance"],
  secops: ["security operations"],
  kb: ["knowledge base"],
  pa: ["performance analytics"],
  va: ["virtual agent"],
  sp: ["service portal"],
  uib: ["ui builder"],
  aes: ["app engine studio"],
  ire: ["identification and reconciliation"],
  awa: ["advanced work assignment"],
  nlu: ["natural language understanding"],
  sso: ["single sign-on"],
  mfa: ["multi-factor authentication"],
  oob: ["out of box", "out-of-box"],
  ootb: ["out of box", "out-of-box"],
};

const API_WORDS = /\b(?:glide[a-z0-9_]*|sn_[a-z0-9_]+|g_[a-z0-9_]+|gs|api|method|parameter|argument|endpoint|script|function)\b/i;

export interface TermGroup {
  /** Token as written in the query (lowercased). */
  term: string;
  /** FTS5 phrase alternatives; the group matches when any alternative matches. */
  alternatives: string[];
}

export interface QueryAnalysis {
  /** Content terms after stopword removal, lowercased, in query order. */
  terms: string[];
  /** One group per content term, including identifier splits and alias/acronym expansions. */
  groups: TermGroup[];
  /** Every word the query refers to, including camelCase splits and aliases; used for identifier boosts. */
  words: Set<string>;
  apiLike: boolean;
  definitionLike: boolean;
}

export function splitIdentifier(identifier: string): string[] {
  return identifier
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .replace(/([a-z\d])([A-Z])/g, "$1 $2")
    .toLowerCase()
    .match(/[a-z0-9]+/g) ?? [];
}

function isIdentifierLike(token: string): boolean {
  return /[a-z][A-Z]/.test(token) || /^[A-Z]{2,}[a-z]/.test(token) || token.includes("_") || /[a-z]\d|\d[a-z]/i.test(token);
}

function quote(phrase: string): string {
  return `"${phrase.replaceAll('"', '""')}"`;
}

export function analyzeQuery(query: string): QueryAnalysis {
  // Dots separate object and method (GlideRecord.addQuery); parentheses and arguments are noise.
  // Letters and digits in any script, matching the full-text index's unicode61 tokenizer.
  const tokens = query.replace(/\([^)]*\)/g, " ").match(/[\p{L}\p{N}_]+/gu) ?? [];
  const groups: TermGroup[] = [];
  const words = new Set<string>();
  const seen = new Set<string>();
  for (const token of tokens) {
    const lower = token.toLowerCase();
    const identifier = isIdentifierLike(token);
    const aliases = IDENTIFIER_ALIASES[lower] ?? [];
    const acronyms = ACRONYMS[lower] ?? [];
    splitIdentifier(token).forEach((word) => words.add(word));
    words.add(lower);
    for (const alias of aliases) {
      words.add(alias.toLowerCase());
      splitIdentifier(alias).forEach((word) => words.add(word));
    }
    if (!identifier && !aliases.length && !acronyms.length && STOPWORDS.has(lower)) continue;
    if (lower.length === 1 && !identifier) continue;
    if (seen.has(lower)) continue;
    seen.add(lower);
    const split = splitIdentifier(token);
    const alternatives = new Set<string>([quote(lower)]);
    if (split.length > 1) alternatives.add(quote(split.join(" ")));
    for (const alias of aliases) {
      alternatives.add(quote(alias.toLowerCase()));
      const aliasSplit = splitIdentifier(alias);
      if (aliasSplit.length > 1) alternatives.add(quote(aliasSplit.join(" ")));
    }
    for (const expansion of acronyms) alternatives.add(quote(expansion));
    groups.push({ term: lower, alternatives: [...alternatives] });
  }
  return {
    terms: groups.map((group) => group.term),
    groups,
    words,
    apiLike: API_WORDS.test(query) || tokens.some(isIdentifierLike),
    definitionLike: /^\s*(?:what\s+(?:is|are)|define|definition\s+of|meaning\s+of)\b/i.test(query) || /\b(?:definition|glossary|meaning)\b/i.test(query),
  };
}

export function groupExpression(group: TermGroup): string {
  return group.alternatives.length === 1 ? group.alternatives[0]! : `(${group.alternatives.join(" OR ")})`;
}

/**
 * Builds progressively relaxed FTS5 expressions. `frequencies` holds each group's matching row count; groups
 * that match nothing are dropped, and each tier removes the next most frequent remaining group, so the rarest
 * (most discriminating) terms are the last to be relaxed. Tier 0 also ORs in the exact phrase so bm25 rewards it.
 */
export function lexicalTiers(analysis: QueryAnalysis, frequencies: number[]): string[] {
  const present = analysis.groups
    .map((group, index) => ({ group, frequency: frequencies[index] ?? 0, index }))
    .filter((item) => item.frequency > 0)
    .sort((left, right) => left.frequency - right.frequency || left.index - right.index);
  if (!present.length) return [];
  const minimum = present.length <= 2 ? 1 : Math.ceil(present.length / 2);
  const tiers: string[] = [];
  for (let size = present.length; size >= minimum; size -= 1) {
    const kept = present.slice(0, size).sort((left, right) => left.index - right.index);
    const conjunction = kept.map((item) => groupExpression(item.group)).join(" AND ");
    if (size === present.length && kept.length > 1) {
      tiers.push(`${quote(kept.map((item) => item.group.term).join(" "))} OR (${conjunction})`);
    } else {
      tiers.push(conjunction);
    }
  }
  return tiers;
}
