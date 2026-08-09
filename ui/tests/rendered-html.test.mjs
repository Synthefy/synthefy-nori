import assert from "node:assert/strict";
import { access, readFile, stat } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders Nori Studio metadata and a useful loading state", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Nori Studio — Explore tabular intelligence<\/title>/i);
  assert.match(html, /Loading the public credit dataset/);
  assert.match(html, /property="og:image" content="http:\/\/localhost(?::3000)?\/og-v2\.png"/i);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("ships the product UI, local CSV importer, public artifact, and social card", async () => {
  const [page, importer, layout, packageJson, readme, dataFile, socialCard, ...starterFiles] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/local-datasets.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../README.md", import.meta.url), "utf8"),
    readFile(new URL("../public/data/nori-embeddings.json", import.meta.url), "utf8"),
    stat(new URL("../public/og-v2.png", import.meta.url)),
    readFile(new URL("../public/data/starters/penguins.csv", import.meta.url), "utf8"),
    readFile(new URL("../public/data/starters/auto-mpg.csv", import.meta.url), "utf8"),
    readFile(new URL("../public/data/starters/restaurant-tips.csv", import.meta.url), "utf8"),
    readFile(new URL("../public/data/starters/titanic.csv", import.meta.url), "utf8"),
  ]);

  assert.match(page, /Embeddings/);
  assert.match(page, /SHAP-IQ interactions/);
  assert.match(page, /Zero-shot inference/);
  assert.match(page, /Scenario inputs/);
  assert.match(page, /Interface preview/);
  assert.match(page, /Add dataset/);
  assert.match(page, /mode-tabs/);
  assert.match(page, /Five ready-to-explore tables/);
  assert.match(page, /Credit Card Default/);
  assert.match(page, /STARTER_DATASETS/);
  assert.match(importer, /Drop a CSV here/);
  assert.match(importer, /Files are parsed locally/);
  assert.match(importer, /Public CSV URL/);
  assert.match(importer, /Pick the target column/);
  assert.match(importer, /Random context \/ query split/);
  assert.match(importer, /prepareDataset/);
  assert.match(layout, /x-forwarded-host/);
  assert.match(layout, /\/og-v2\.png/);
  assert.match(packageJson, /"name": "nori-studio-ui"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.match(readme, /UCI Default of Credit Card Clients/);
  assert.equal(JSON.parse(dataFile).n, 3000);
  assert.ok(socialCard.size > 100_000);
  assert.equal(starterFiles.length, 4);
  starterFiles.forEach((csv) => assert.ok(csv.split("\n").length > 200));

  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});
