import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host = requestHeaders.get("x-forwarded-host")?.split(",")[0]?.trim()
    ?? requestHeaders.get("host")
    ?? "localhost:3000";
  const protocol = requestHeaders.get("x-forwarded-proto")?.split(",")[0]?.trim()
    ?? (host.startsWith("localhost") ? "http" : "https");
  let origin = "http://localhost:3000";
  try {
    origin = new URL(`${protocol}://${host}`).origin;
  } catch {
    // Keep a valid metadata URL even if a development proxy sends a malformed host.
  }
  const socialImage = `${origin}/og.png`;

  return {
    title: "Nori Studio — Explore tabular intelligence",
    description: "Explore Nori embeddings, interpretability, zero-shot inference, and scenario simulation on a public credit dataset.",
    openGraph: {
      title: "Nori Studio — See what Nori understands",
      description: "An interactive workspace for target-aware tabular intelligence.",
      type: "website",
      images: [{ url: socialImage, width: 1536, height: 1024, alt: "Nori Studio target-aware embedding map" }],
    },
    twitter: {
      card: "summary_large_image",
      title: "Nori Studio — See what Nori understands",
      description: "An interactive workspace for target-aware tabular intelligence.",
      images: [socialImage],
    },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
