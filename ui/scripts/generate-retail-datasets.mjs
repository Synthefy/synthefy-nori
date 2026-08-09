import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDirectory = resolve(root, "public/data/retail");

function randomFactory(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(random) {
  const left = Math.max(random(), 1e-9);
  const right = Math.max(random(), 1e-9);
  return Math.sqrt(-2 * Math.log(left)) * Math.cos(2 * Math.PI * right);
}

const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const logistic = (value) => 1 / (1 + Math.exp(-value));
const pick = (random, values) => values[Math.floor(random() * values.length)];
const yesNo = (value) => value ? "Yes" : "No";
const round = (value, digits = 0) => Number(value.toFixed(digits));

function csvEscape(value) {
  const text = String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function toCSV(rows) {
  const headers = Object.keys(rows[0]);
  return [headers.join(","), ...rows.map((row) => headers.map((header) => csvEscape(row[header])).join(","))].join("\n") + "\n";
}

function lifetimeValueRows() {
  const random = randomFactory(1101);
  const channels = ["Organic", "Paid social", "Search", "Referral", "Store"];
  const regions = ["Northeast", "South", "Midwest", "West"];
  return Array.from({ length: 1_050 }, (_, index) => {
    const tenure = 2 + Math.floor(random() * 70);
    const affinity = clamp(normal(random) * 0.75 + tenure / 52, -1.5, 2.8);
    const orders = Math.max(1, Math.round(3.8 + affinity * 3.1 + normal(random) * 2.3));
    const averageOrderValue = clamp(48 + affinity * 18 + normal(random) * 23, 12, 210);
    const returnRate = clamp(0.15 - affinity * 0.025 + normal(random) * 0.055, 0, 0.42);
    const emailClickRate = clamp(0.16 + affinity * 0.07 + normal(random) * 0.08, 0, 0.66);
    const loyaltyPoints = Math.max(0, Math.round(orders * averageOrderValue * (0.6 + random() * 0.6)));
    const supportTickets = Math.max(0, Math.round(1.3 - affinity * 0.28 + normal(random) * 0.9));
    const value = Math.max(20, orders * averageOrderValue * (1 - returnRate) * (1.06 + tenure / 120) + loyaltyPoints * 0.08 + normal(random) * 80);
    return {
      customer_id: `C${String(index + 1).padStart(5, "0")}`,
      tenure_months: tenure,
      orders_12m: orders,
      avg_order_value: round(averageOrderValue, 2),
      return_rate: round(returnRate, 3),
      email_click_rate: round(emailClickRate, 3),
      loyalty_points: loyaltyPoints,
      support_tickets: supportTickets,
      acquisition_channel: pick(random, channels),
      region: pick(random, regions),
      lifetime_value_12m: round(value, 2),
    };
  });
}

function churnRows() {
  const random = randomFactory(2202);
  const categories = ["Beauty", "Home", "Apparel", "Electronics", "Grocery"];
  return Array.from({ length: 1_000 }, (_, index) => {
    const tenure = 1 + Math.floor(random() * 64);
    const recency = Math.max(1, Math.round(16 + Math.abs(normal(random)) * 30 + random() * 38));
    const orders = Math.max(0, Math.round(5.5 - recency / 28 + tenure / 30 + normal(random) * 2.1));
    const basket = clamp(58 + normal(random) * 27, 10, 190);
    const discountShare = clamp(0.28 + normal(random) * 0.18, 0, 0.9);
    const returns = Math.max(0, Math.round(normal(random) * 0.8 + discountShare * 2.1));
    const tickets = Math.max(0, Math.round(normal(random) * 0.9 + returns * 0.45));
    const loyalty = random() < logistic(-0.25 + tenure / 22 + orders / 7);
    const risk = logistic(-2.1 + recency / 35 - orders * 0.25 + discountShare * 1.2 + tickets * 0.36 - (loyalty ? 0.8 : 0));
    return {
      customer_id: `R${String(index + 1).padStart(5, "0")}`,
      tenure_months: tenure,
      days_since_purchase: recency,
      orders_90d: orders,
      avg_basket: round(basket, 2),
      discount_share: round(discountShare, 3),
      returns_90d: returns,
      support_tickets: tickets,
      loyalty_member: yesNo(loyalty),
      preferred_category: pick(random, categories),
      churned_90d: random() < risk ? 1 : 0,
    };
  });
}

function conversionRows() {
  const random = randomFactory(3303);
  const channels = ["Organic", "Paid social", "Search", "Email", "Affiliate"];
  const devices = ["Mobile", "Desktop", "Tablet"];
  return Array.from({ length: 1_200 }, (_, index) => {
    const sessions = 1 + Math.floor(random() * 14);
    const views = Math.max(sessions, Math.round(sessions * (2.2 + random() * 3.2) + normal(random) * 3));
    const carts = Math.max(0, Math.round(views * clamp(0.05 + random() * 0.16, 0, 0.35)));
    const wishlist = Math.max(0, Math.round(normal(random) + carts * 0.45));
    const minutes = clamp(2.3 + sessions * 0.45 + carts * 0.8 + normal(random) * 1.8, 0.4, 19);
    const priorOrders = Math.max(0, Math.round(normal(random) * 1.4 + random() * 4));
    const emailClicks = Math.max(0, Math.round(random() * 5 - 1));
    const channel = pick(random, channels);
    const probability = logistic(-3.2 + carts * 0.72 + wishlist * 0.2 + minutes * 0.12 + priorOrders * 0.27 + emailClicks * 0.16 + (channel === "Email" ? 0.35 : 0));
    return {
      visitor_id: `V${String(index + 1).padStart(5, "0")}`,
      sessions_30d: sessions,
      product_views: views,
      cart_adds: carts,
      wishlist_items: wishlist,
      avg_session_minutes: round(minutes, 2),
      prior_orders: priorOrders,
      email_clicks_30d: emailClicks,
      acquisition_channel: channel,
      device: pick(random, devices),
      converted_14d: random() < probability ? 1 : 0,
    };
  });
}

function promotionRows() {
  const random = randomFactory(4404);
  const tiers = ["None", "Silver", "Gold", "Platinum"];
  const categories = ["Beauty", "Home", "Apparel", "Electronics", "Grocery"];
  return Array.from({ length: 1_300 }, (_, index) => {
    const treated = random() < 0.5;
    const recency = 2 + Math.floor(random() * 110);
    const orders = Math.max(0, Math.round(4.2 - recency / 35 + random() * 5 + normal(random)));
    const basket = clamp(62 + normal(random) * 29, 9, 220);
    const fullPriceShare = clamp(0.48 + normal(random) * 0.23, 0.02, 1);
    const engagement = clamp(0.2 + normal(random) * 0.13, 0, 0.75);
    const tier = pick(random, tiers);
    const discount = treated ? pick(random, [10, 15, 20, 25]) : 0;
    const priceSensitive = 1 - fullPriceShare;
    const treatmentLift = treated ? 0.2 + priceSensitive * 1.4 + Math.min(recency / 120, 0.8) + discount / 55 : 0;
    const probability = logistic(-2.6 + orders * 0.22 + engagement * 2.2 - recency / 95 + (tier === "Gold" || tier === "Platinum" ? 0.45 : 0) + treatmentLift);
    return {
      customer_id: `P${String(index + 1).padStart(5, "0")}`,
      promotion_received: treated ? 1 : 0,
      discount_pct: discount,
      days_since_purchase: recency,
      orders_6m: orders,
      avg_basket: round(basket, 2),
      full_price_share: round(fullPriceShare, 3),
      email_engagement: round(engagement, 3),
      loyalty_tier: tier,
      preferred_category: pick(random, categories),
      purchased_30d: random() < probability ? 1 : 0,
    };
  });
}

function campaignRows() {
  const random = randomFactory(5505);
  const segments = ["Young household", "Family", "Urban single", "Established", "Retired"];
  const categories = ["Beauty", "Home", "Apparel", "Electronics", "Grocery"];
  const channels = ["Email", "SMS", "Paid social", "Direct mail"];
  return Array.from({ length: 1_100 }, (_, index) => {
    const recency = 1 + Math.floor(random() * 150);
    const frequency = Math.max(0, Math.round(7 - recency / 32 + random() * 9 + normal(random) * 1.8));
    const monetary = Math.max(0, frequency * clamp(58 + normal(random) * 24, 12, 180));
    const openRate = clamp(0.22 + normal(random) * 0.16, 0, 0.82);
    const sms = random() < 0.44;
    const webVisits = Math.max(0, Math.round(frequency * 0.7 + random() * 7 + normal(random) * 1.2));
    const channel = pick(random, channels);
    const probability = logistic(-3 + frequency * 0.14 + Math.log1p(monetary) * 0.2 + openRate * 2.5 + webVisits * 0.08 - recency / 120 + (sms && channel === "SMS" ? 0.4 : 0));
    return {
      customer_id: `M${String(index + 1).padStart(5, "0")}`,
      days_since_purchase: recency,
      orders_12m: frequency,
      spend_12m: round(monetary, 2),
      email_open_rate: round(openRate, 3),
      sms_opt_in: yesNo(sms),
      web_visits_30d: webVisits,
      household_segment: pick(random, segments),
      preferred_category: pick(random, categories),
      campaign_channel: channel,
      responded_30d: random() < probability ? 1 : 0,
    };
  });
}

await mkdir(outputDirectory, { recursive: true });
const datasets = [
  ["customer-lifetime-value.csv", lifetimeValueRows()],
  ["customer-churn.csv", churnRows()],
  ["customer-conversion.csv", conversionRows()],
  ["promotion-uplift.csv", promotionRows()],
  ["campaign-response.csv", campaignRows()],
];

for (const [filename, rows] of datasets) {
  await writeFile(resolve(outputDirectory, filename), toCSV(rows), "utf8");
}

console.log(`Generated ${datasets.length} retail demo datasets in ${outputDirectory}`);
