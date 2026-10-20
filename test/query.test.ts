import { describe, expect, it } from "vitest";
import { analyzeQuery, groupExpression, lexicalTiers, splitIdentifier } from "../src/query.js";

describe("analyzeQuery", () => {
  it("drops stopwords but keeps content terms in order", () => {
    expect(analyzeQuery("how do I create a business rule that runs after insert").terms).toEqual(["create", "business", "rule", "runs", "insert"]);
  });

  it("never drops identifiers even when they look like stopwords", () => {
    expect(analyzeQuery("gs.getProperty").terms).toEqual(["gs", "getproperty"]);
    expect(analyzeQuery("what is g_form").terms).toEqual(["g_form"]);
  });

  it("adds camelCase splits and object aliases as alternatives", () => {
    const [objectGroup, methodGroup] = analyzeQuery("gs.getProperty").groups;
    expect(objectGroup?.alternatives).toEqual(expect.arrayContaining(['"gs"', '"glidesystem"', '"glide system"']));
    expect(methodGroup?.alternatives).toEqual(['"getproperty"', '"get property"']);
  });

  it("collects identifier words for exact API boosts", () => {
    const { words } = analyzeQuery("gs.getProperty");
    expect([...words]).toEqual(expect.arrayContaining(["get", "property", "getproperty", "glidesystem", "glide", "system"]));
  });

  it("expands ServiceNow acronyms on the lexical leg", () => {
    const groups = analyzeQuery("ATF test for a BR").groups;
    expect(groups.map((group) => group.term)).toEqual(["atf", "test", "br"]);
    expect(groups[0]?.alternatives).toContain('"automated test framework"');
    expect(groups[2]?.alternatives).toContain('"business rule"');
  });

  it("ignores call arguments and quotes FTS phrases safely", () => {
    expect(analyzeQuery("setWorkflow(false)").terms).toEqual(["setworkflow"]);
    expect(groupExpression({ term: 'a"b', alternatives: ['"a""b"'] })).toBe('"a""b"');
  });

  it("classifies API-like and definition-like queries", () => {
    expect(analyzeQuery("GlideRecord addQuery").apiLike).toBe(true);
    expect(analyzeQuery("create an incident").apiLike).toBe(false);
    expect(analyzeQuery("what is dot-walking").definitionLike).toBe(true);
  });
});

describe("lexicalTiers", () => {
  it("starts with phrase-or-conjunction and relaxes the most frequent terms first", () => {
    const analysis = analyzeQuery("delete records query");
    expect(lexicalTiers(analysis, [50, 5000, 900])).toEqual([
      '"delete records query" OR ("delete" AND "records" AND "query")',
      '"delete" AND "query"',
    ]);
  });

  it("drops terms that match nothing and returns no tiers when nothing matches", () => {
    const analysis = analyzeQuery("glidrecord addquery");
    expect(lexicalTiers(analysis, [0, 12])).toEqual(['"addquery"']);
    expect(lexicalTiers(analysis, [0, 0])).toEqual([]);
  });
});

describe("splitIdentifier", () => {
  it("splits camelCase, acronyms, and underscores", () => {
    expect(splitIdentifier("GlideRecordSecure")).toEqual(["glide", "record", "secure"]);
    expect(splitIdentifier("RESTMessageV2")).toEqual(["rest", "message", "v2"]);
    expect(splitIdentifier("sys_id")).toEqual(["sys", "id"]);
  });
});

describe("analyzeQuery tokenization", () => {
  it("keeps words in any script, as the full-text index does", () => {
    expect(analyzeQuery("Überblick über Zähler").terms).toEqual(["überblick", "über", "zähler"]);
  });
});
