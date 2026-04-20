#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { Presentation, PresentationFile } from "@oai/artifact-tool";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");
const PRESENTATION_DIR = path.join(ROOT, "Final Project Presentation");
const OUTPUT_PPTX = path.join(PRESENTATION_DIR, "final_presentation.pptx");
const SCRIPT_MD = path.join(PRESENTATION_DIR, "final_presentation_script.md");

const SLIDE_W = 1280;
const SLIDE_H = 720;

const COLORS = {
  bg: "#08131d",
  bgAlt: "#0f1d2b",
  panel: "#112438",
  panelSoft: "#182e45",
  panelLight: "#eef4fb",
  ink: "#0e1722",
  inkSoft: "#314557",
  white: "#f7fbff",
  muted: "#b4c5d8",
  mutedDark: "#6b7d8f",
  cyan: "#7fe7ff",
  orange: "#ff9c49",
  green: "#5de2a5",
  red: "#ff6a7a",
  border: "#29435c",
  borderLight: "#d6e0ea",
};

const FONT = {
  title: "Aptos Display",
  body: "Aptos",
};

const IMAGES = {
  title: path.join(PRESENTATION_DIR, "assets_foot_in_door_security.png"),
  concept: path.join(PRESENTATION_DIR, "assets_concept_trojan_horse.png"),
  setup: path.join(PRESENTATION_DIR, "assets_plan_blueprint_schematic.png"),
  conclusion: path.join(PRESENTATION_DIR, "assets_conclusion_secure_horizon.png"),
};

const FIGURES = {
  asr: path.join(PRESENTATION_DIR, "figures", "figure_asr_by_model.png"),
  qwen: path.join(PRESENTATION_DIR, "figures", "figure_refusal_rate_by_model.png"),
};

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

async function loadSpeakerNotes(scriptPath) {
  const text = await fs.readFile(scriptPath, "utf8");
  const regex = /## Slide (\d+):[^\n]*\n\*\*Presenter:\*\*.*?\n\*\*Target time:\*\*.*?\n\n\*\*Script\*\*\n\n"([\s\S]*?)"\n(?=\n---|\s*$)/g;
  const notes = new Map();
  let match;
  while ((match = regex.exec(text)) !== null) {
    const slideNumber = Number(match[1]);
    const speakerText = match[2].replace(/\n+/g, " ").trim();
    notes.set(slideNumber, speakerText);
  }
  if (notes.size !== 10) {
    throw new Error(`Expected 10 script note blocks, found ${notes.size}`);
  }
  return notes;
}

function addText(slide, {
  text,
  left,
  top,
  width,
  height,
  fontSize = 24,
  color = COLORS.white,
  bold = false,
  typeface = FONT.body,
  fill = null,
  lineColor = null,
  radius = "rect",
  align = "left",
  valign = "top",
  insets = { left: 16, right: 16, top: 12, bottom: 12 },
}) {
  const shape = slide.shapes.add({
    geometry: radius,
    position: { left, top, width, height },
  });
  if (fill) {
    shape.fill.color = fill;
  }
  if (lineColor) {
    shape.line.color = lineColor;
    shape.line.width = 1.2;
  } else {
    shape.line.visible = false;
  }
  shape.text = text;
  shape.text.fontSize = fontSize;
  shape.text.color = color;
  shape.text.bold = bold;
  shape.text.typeface = typeface;
  shape.text.alignment = align;
  shape.text.verticalAlignment = valign;
  shape.text.insets = insets;
  shape.text.wrap = true;
  return shape;
}

function addPill(slide, label, left, top, color) {
  const width = Math.max(140, Math.min(320, 48 + label.length * 12));
  return addText(slide, {
    text: label,
    left,
    top,
    width,
    height: 38,
    fontSize: 14,
    bold: true,
    color: COLORS.bg,
    fill: color,
    radius: "roundRect",
    align: "center",
    valign: "middle",
    insets: { left: 10, right: 10, top: 6, bottom: 6 },
  });
}

function addSlideFrame(slide, { number, dark = true }) {
  slide.background.fill = dark ? COLORS.bg : "#f5f9fd";
  addText(slide, {
    text: "CSCI/DASC 6040 FINAL PROJECT",
    left: 56,
    top: 24,
    width: 420,
    height: 28,
    fontSize: 13,
    bold: true,
    color: dark ? COLORS.cyan : COLORS.inkSoft,
    fill: null,
    align: "left",
  });
  addText(slide, {
    text: `${number}`,
    left: 1182,
    top: 24,
    width: 50,
    height: 28,
    fontSize: 13,
    bold: true,
    color: dark ? COLORS.muted : COLORS.mutedDark,
    fill: null,
    align: "right",
  });
}

function addTitle(slide, { kicker, title, subtitle, dark = true }) {
  if (kicker) {
    addPill(slide, kicker.toUpperCase(), 56, 70, dark ? COLORS.orange : COLORS.cyan);
  }
  addText(slide, {
    text: title,
    left: 56,
    top: kicker ? 118 : 76,
    width: 700,
    height: 88,
    fontSize: 32,
    bold: true,
    typeface: FONT.title,
    color: dark ? COLORS.white : COLORS.ink,
    fill: null,
  });
  if (subtitle) {
    addText(slide, {
      text: subtitle,
      left: 56,
      top: kicker ? 198 : 148,
      width: 760,
      height: 62,
      fontSize: 18,
      color: dark ? COLORS.muted : COLORS.mutedDark,
      fill: null,
    });
  }
}

function addCard(slide, {
  left,
  top,
  width,
  height,
  title,
  body,
  accent = COLORS.cyan,
  dark = false,
}) {
  const fill = dark ? COLORS.panel : "#ffffff";
  const bodyColor = dark ? COLORS.muted : COLORS.inkSoft;
  addText(slide, {
    text: "",
    left,
    top,
    width,
    height,
    fill,
    lineColor: dark ? COLORS.border : COLORS.borderLight,
    radius: "roundRect",
  });
  addText(slide, {
    text: title,
    left: left + 16,
    top: top + 14,
    width: width - 32,
    height: 40,
    fontSize: 22,
    bold: true,
    typeface: FONT.title,
    color: accent,
    fill: null,
  });
  addText(slide, {
    text: body,
    left: left + 16,
    top: top + 52,
    width: width - 32,
    height: height - 64,
    fontSize: 17,
    color: bodyColor,
    fill: null,
  });
}

function addBulletList(slide, {
  items,
  left,
  top,
  width,
  height,
  fontSize = 22,
  color = COLORS.white,
  dark = true,
}) {
  const text = items.map((item) => `• ${item}`).join("\n");
  return addText(slide, {
    text,
    left,
    top,
    width,
    height,
    fontSize,
    color,
    fill: null,
    insets: { left: 6, right: 6, top: 4, bottom: 4 },
  });
}

function addNativeBarChart(slide, {
  left,
  top,
  width,
  height,
  categories,
  series,
  ymax,
  title,
  note,
}) {
  addText(slide, {
    text: "",
    left,
    top,
    width,
    height,
    fill: "#ffffff",
    lineColor: COLORS.borderLight,
    radius: "roundRect",
  });

  const chart = slide.charts.add("bar");
  chart.position = { left: left + 20, top: top + 56, width: width - 40, height: height - 96 };
  chart.title = title;
  chart.titleTextStyle.typeface = FONT.title;
  chart.titleTextStyle.fontSize = 20;
  chart.titleTextStyle.bold = true;
  chart.titleTextStyle.color = COLORS.ink;
  chart.categories = categories;
  chart.hasLegend = true;
  chart.legend.position = "bottom";
  chart.legend.textStyle.typeface = FONT.body;
  chart.legend.textStyle.fontSize = 14;
  chart.legend.textStyle.color = COLORS.inkSoft;
  chart.dataLabels.showValue = true;
  chart.dataLabels.position = "outEnd";
  chart.barOptions.direction = "column";
  chart.xAxis.textStyle.typeface = FONT.body;
  chart.xAxis.textStyle.fontSize = 13;
  chart.xAxis.textStyle.color = COLORS.inkSoft;
  chart.xAxis.line.color = COLORS.borderLight;
  chart.xAxis.line.width = 1;
  chart.yAxis.title = "ASR (%)";
  chart.yAxis.min = 0;
  chart.yAxis.max = ymax;
  chart.yAxis.majorUnit = ymax / 4;
  chart.yAxis.textStyle.typeface = FONT.body;
  chart.yAxis.textStyle.fontSize = 13;
  chart.yAxis.textStyle.color = COLORS.inkSoft;
  chart.yAxis.line.color = COLORS.borderLight;
  chart.yAxis.line.width = 1;
  chart.yAxis.majorGridlines.color = COLORS.borderLight;
  chart.yAxis.majorGridlines.width = 1;

  for (const item of series) {
    const chartSeries = chart.series.add(item.name);
    chartSeries.values = item.values;
    chartSeries.categories = categories;
    chartSeries.fill = item.color;
    chartSeries.stroke.fill = item.color;
    chartSeries.stroke.width = 1.4;
  }

  addText(slide, {
    text: note,
    left: left + 20,
    top: top + height - 34,
    width: width - 40,
    height: 24,
    fontSize: 12,
    color: COLORS.mutedDark,
    fill: null,
  });
}

function addFigurePanel(slide, {
  blob,
  left,
  top,
  width,
  height,
  alt,
}) {
  addText(slide, {
    text: "",
    left,
    top,
    width,
    height,
    fill: "#ffffff",
    lineColor: COLORS.borderLight,
    radius: "roundRect",
  });
  const image = slide.images.add({ blob, fit: "contain", alt });
  image.position = { left: left + 14, top: top + 14, width: width - 28, height: height - 28 };
  return image;
}

async function buildDeck() {
  const speakerNotes = await loadSpeakerNotes(SCRIPT_MD);
  const deck = Presentation.create({
    slideSize: { width: SLIDE_W, height: SLIDE_H },
  });

  const titleBlob = await readImageBlob(IMAGES.title);
  const conceptBlob = await readImageBlob(IMAGES.concept);
  const setupBlob = await readImageBlob(IMAGES.setup);
  const conclusionBlob = await readImageBlob(IMAGES.conclusion);
  const asrFigureBlob = await readImageBlob(FIGURES.asr);
  const qwenFigureBlob = await readImageBlob(FIGURES.qwen);

  const slide1 = deck.slides.add();
  addSlideFrame(slide1, { number: 1, dark: true });
  const hero = slide1.images.add({ blob: titleBlob, fit: "cover", alt: "Abstract cybersecurity illustration" });
  hero.position = { left: 780, top: 84, width: 430, height: 510 };
  addText(slide1, {
    text: "",
    left: 770,
    top: 74,
    width: 450,
    height: 530,
    fill: null,
    lineColor: COLORS.border,
    radius: "roundRect",
  });
  addPill(slide1, "Reproduction Study", 56, 92, COLORS.cyan);
  addText(slide1, {
    text: "Reproducing \"Foot-In-The-Door\": Multi-turn Model Jailbreaking",
    left: 56,
    top: 146,
    width: 640,
    height: 178,
    fontSize: 38,
    bold: true,
    typeface: FONT.title,
    color: COLORS.white,
    fill: null,
  });
  addText(slide1, {
    text: "Matthew Aiken and Chris Murphy\nEMNLP 2025 reproduction study for CSCI/DASC 6040",
    left: 56,
    top: 330,
    width: 560,
    height: 88,
    fontSize: 22,
    color: COLORS.muted,
    fill: null,
  });
  addPill(slide1, "Bottom line", 56, 442, COLORS.orange);
  addCard(slide1, {
    left: 56,
    top: 490,
    width: 300,
    height: 118,
    title: "Observed in follow-up",
    body: "The GPU and vLLM follow-up produced harmful outputs on some models.",
    accent: COLORS.cyan,
    dark: true,
  });
  addCard(slide1, {
    left: 376,
    top: 490,
    width: 300,
    height: 118,
    title: "Core result",
    body: "The paper's Qwen-family FITD result was still not cleanly reproduced.",
    accent: COLORS.orange,
    dark: true,
  });
  slide1.speakerNotes.setText(speakerNotes.get(1));

  const slide2 = deck.slides.add();
  addSlideFrame(slide2, { number: 2, dark: false });
  addTitle(slide2, {
    kicker: "Paper Claim",
    title: "What the Paper Claims",
    subtitle: "The paper argues that conversation path, not just the final prompt, can change whether a model refuses or complies.",
    dark: false,
  });
  addText(slide2, {
    text: "",
    left: 816,
    top: 106,
    width: 416,
    height: 492,
    fill: null,
    lineColor: COLORS.borderLight,
    radius: "roundRect",
  });
  const conceptImg = slide2.images.add({ blob: conceptBlob, fit: "cover", alt: "Trojan horse metaphor" });
  conceptImg.position = { left: 828, top: 118, width: 392, height: 468 };
  addCard(slide2, {
    left: 56,
    top: 248,
    width: 330,
    height: 230,
    title: "Direct harmful prompt",
    body: "Ask for the harmful action immediately.\n\nExpected behavior: refusal.",
    accent: COLORS.cyan,
    dark: false,
  });
  addCard(slide2, {
    left: 414,
    top: 248,
    width: 330,
    height: 230,
    title: "FITD multi-turn path",
    body: "Start with harmless-seeming security or manipulation questions, then gradually escalate to the harmful request.",
    accent: COLORS.orange,
    dark: false,
  });
  addText(slide2, {
    text: "If the escalation path works reliably, safety depends on the entire conversation history rather than only the final turn.",
    left: 56,
    top: 520,
    width: 688,
    height: 92,
    fontSize: 22,
    color: COLORS.ink,
    bold: true,
    fill: "#e8f6fb",
    radius: "roundRect",
  });
  slide2.speakerNotes.setText(speakerNotes.get(2));

  const slide3 = deck.slides.add();
  addSlideFrame(slide3, { number: 3, dark: true });
  addTitle(slide3, {
    kicker: "Research Questions",
    title: "What We Tested",
    subtitle: "We treated this as a reproduction study first and an interpretation study second.",
    dark: true,
  });
  addCard(slide3, {
    left: 56,
    top: 244,
    width: 356,
    height: 268,
    title: "Q1",
    body: "Can we reproduce a standard-vs-FITD gap on real harmful prompts?",
    accent: COLORS.cyan,
    dark: true,
  });
  addCard(slide3, {
    left: 462,
    top: 244,
    width: 356,
    height: 268,
    title: "Q2",
    body: "Does a vigilant defense prompt reduce the effect once FITD is applied?",
    accent: COLORS.orange,
    dark: true,
  });
  addCard(slide3, {
    left: 868,
    top: 244,
    width: 356,
    height: 268,
    title: "Q3",
    body: "Does a closer GPU and vLLM runtime reveal harmful behavior that the initial local pipeline missed?",
    accent: COLORS.green,
    dark: true,
  });
  addText(slide3, {
    text: "We aimed for rigorous evidence even if the outcome stayed partial or negative.",
    left: 170,
    top: 560,
    width: 940,
    height: 72,
    fontSize: 24,
    color: COLORS.bg,
    bold: true,
    fill: COLORS.green,
    radius: "roundRect",
    align: "center",
    valign: "middle",
  });
  slide3.speakerNotes.setText(speakerNotes.get(3));

  const slide4 = deck.slides.add();
  addSlideFrame(slide4, { number: 4, dark: false });
  addTitle(slide4, {
    kicker: "Method",
    title: "Experimental Design",
    subtitle: "The final evidence combines an initial local scaffold evaluation with a later GPU and vLLM follow-up.",
    dark: false,
  });
  const setupImg = slide4.images.add({ blob: setupBlob, fit: "cover", alt: "Blueprint-style setup visual" });
  setupImg.position = { left: 900, top: 126, width: 300, height: 188 };
  addCard(slide4, {
    left: 56,
    top: 238,
    width: 360,
    height: 184,
    title: "Phase 1: local scaffold work",
    body: "Qwen 2.5 3B, Gemma 4, local Llama 3, and an exact-model Qwen2-7B author-chain check on a 10-example jailbreakbench slice.",
    accent: COLORS.cyan,
    dark: false,
  });
  addCard(slide4, {
    left: 448,
    top: 238,
    width: 360,
    height: 184,
    title: "Phase 2: AdvBench Matrix",
    body: "First 25 AdvBench examples, five models, three conditions, and a local Qwen 2.5 7B judge served through vLLM.",
    accent: COLORS.orange,
    dark: false,
  });
  addCard(slide4, {
    left: 840,
    top: 238,
    width: 360,
    height: 184,
    title: "Closest feasible match",
    body: "We matched runtime and exact Qwen-family targets more closely, while the adaptive loop and original judge stack remained approximate.",
    accent: COLORS.red,
    dark: false,
  });
  addBulletList(slide4, {
    items: [
      "Three conditions in every matrix cell: Standard, FITD, FITD + Vigilant",
      "The later GPU and vLLM follow-up produced harmful outputs that were largely absent in the initial local runs",
      "The cleanest next Qwen check is still the author-chain path with the raw target preserved",
    ],
    left: 72,
    top: 468,
    width: 1128,
    height: 160,
    fontSize: 22,
    color: COLORS.ink,
    dark: false,
  });
  slide4.speakerNotes.setText(speakerNotes.get(4));

  const slide5 = deck.slides.add();
  addSlideFrame(slide5, { number: 5, dark: true });
  addTitle(slide5, {
    kicker: "Result 1",
    title: "Why a Runtime Follow-up Was Necessary",
    subtitle: "The initial local runs were highly refusal-dominant, so we added a closer runtime check to test whether the pipeline was missing harmful behavior.",
    dark: true,
  });
  addCard(slide5, {
    left: 70,
    top: 260,
    width: 300,
    height: 180,
    title: "Initial local runs",
    body: "Across several sampled conditions, the initial local system produced almost no harmful outputs.",
    accent: COLORS.cyan,
    dark: true,
  });
  addCard(slide5, {
    left: 490,
    top: 260,
    width: 300,
    height: 180,
    title: "GPU/vLLM follow-up",
    body: "The later GPU and vLLM evaluation produced clear harmful responses on some models.",
    accent: COLORS.orange,
    dark: true,
  });
  addCard(slide5, {
    left: 910,
    top: 260,
    width: 300,
    height: 180,
    title: "Qwen-family caveat",
    body: "Some judged Qwen positives still needed manual review before they could be interpreted confidently.",
    accent: COLORS.red,
    dark: true,
  });
  addText(slide5, {
    text: "→",
    left: 390,
    top: 304,
    width: 70,
    height: 90,
    fontSize: 56,
    bold: true,
    color: COLORS.muted,
    fill: null,
    align: "center",
  });
  addText(slide5, {
    text: "→",
    left: 810,
    top: 304,
    width: 70,
    height: 90,
    fontSize: 56,
    bold: true,
    color: COLORS.muted,
    fill: null,
    align: "center",
  });
  addText(slide5, {
    text: "The follow-up made the evaluation more informative, but it still did not yield a clean Qwen-family reproduction.",
    left: 118,
    top: 520,
    width: 1044,
    height: 88,
    fontSize: 24,
    color: COLORS.bg,
    bold: true,
    fill: COLORS.orange,
    radius: "roundRect",
    align: "center",
    valign: "middle",
  });
  slide5.speakerNotes.setText(speakerNotes.get(5));

  const slide6 = deck.slides.add();
  addSlideFrame(slide6, { number: 6, dark: false });
  addTitle(slide6, {
    kicker: "Result 2",
    title: "GPU/vLLM Follow-up: Multi-Model Results",
    subtitle: "These are judge-rescored rates from the GPU/vLLM follow-up, not fully manual-audited exploit totals.",
    dark: false,
  });
  addFigurePanel(slide6, {
    blob: asrFigureBlob,
    left: 56,
    top: 208,
    width: 1168,
    height: 356,
    alt: "Multi-model ASR figure for GPU/vLLM follow-up",
  });
  addText(slide6, {
    text: "Most visible lift: Mistral rose from 48% under standard prompting to 72% under FITD, then declined to 52% under the vigilant condition.",
    left: 102,
    top: 594,
    width: 1076,
    height: 72,
    fontSize: 22,
    color: COLORS.bg,
    bold: true,
    fill: COLORS.green,
    radius: "roundRect",
    align: "center",
    valign: "middle",
  });
  slide6.speakerNotes.setText(speakerNotes.get(6));

  const slide7 = deck.slides.add();
  addSlideFrame(slide7, { number: 7, dark: false });
  addTitle(slide7, {
    kicker: "Result 3",
    title: "Exact Qwen-Family Results Stayed Mixed",
    subtitle: "The exact paper-family Qwen targets still did not show a strong, clean FITD lift in the GPU/vLLM follow-up.",
    dark: false,
  });
  addFigurePanel(slide7, {
    blob: qwenFigureBlob,
    left: 56,
    top: 208,
    width: 760,
    height: 356,
    alt: "Qwen-family ASR figure for GPU/vLLM follow-up",
  });
  addCard(slide7, {
    left: 846,
    top: 228,
    width: 350,
    height: 136,
    title: "Qwen2 7B",
    body: "Standard 12%, FITD 12%, FITD+V 8%. No strong FITD lift.",
    accent: COLORS.cyan,
    dark: false,
  });
  addCard(slide7, {
    left: 846,
    top: 382,
    width: 350,
    height: 136,
    title: "Qwen1.5 7B",
    body: "Standard 12%, FITD 4%, FITD+V 4%. Again, no clear FITD win.",
    accent: COLORS.orange,
    dark: false,
  });
  addText(slide7, {
    text: "Manual-audit note: some judged Qwen positives overlapped with refusal-style outputs on spot-check, so these should be read as preliminary judged rates rather than final audited exploit counts.",
    left: 828,
    top: 538,
    width: 386,
    height: 116,
    fontSize: 16,
    color: COLORS.ink,
    fill: "#fff1e8",
    lineColor: COLORS.orange,
    radius: "roundRect",
  });
  slide7.speakerNotes.setText(speakerNotes.get(7));

  const slide8 = deck.slides.add();
  addSlideFrame(slide8, { number: 8, dark: true });
  addTitle(slide8, {
    kicker: "Interpretation",
    title: "Remaining Reproduction Gaps",
    subtitle: "The GPU/vLLM follow-up reduced one major gap, but some paper-fidelity pieces still remained approximate in this project setting.",
    dark: true,
  });
  addCard(slide8, {
    left: 70,
    top: 252,
    width: 520,
    height: 130,
    title: "Runtime stack",
    body: "The later evaluation used vLLM on a GPU, which is materially closer to the paper than the earlier HF CPU path.",
    accent: COLORS.cyan,
    dark: true,
  });
  addCard(slide8, {
    left: 690,
    top: 252,
    width: 520,
    height: 130,
    title: "Prompt pathway",
    body: "The follow-up used scaffold FITD on AdvBench as the closest feasible approximation, rather than the narrower author raw-target Qwen pathway.",
    accent: COLORS.orange,
    dark: true,
  });
  addCard(slide8, {
    left: 70,
    top: 420,
    width: 520,
    height: 130,
    title: "Judge configuration",
    body: "The follow-up used a local Qwen 2.5 judge, and manual spot-checking showed that some refusal-style outputs still needed audit.",
    accent: COLORS.green,
    dark: true,
  });
  addCard(slide8, {
    left: 690,
    top: 420,
    width: 520,
    height: 130,
    title: "Adaptive attack loop",
    body: "The adaptive assistant-driven FITD loop remained an approximation target for future work rather than an exact one-for-one match in this project.",
    accent: COLORS.red,
    dark: true,
  });
  slide8.speakerNotes.setText(speakerNotes.get(8));

  const slide9 = deck.slides.add();
  addSlideFrame(slide9, { number: 9, dark: true });
  const concl = slide9.images.add({ blob: conclusionBlob, fit: "cover", alt: "Secure horizon image" });
  concl.position = { left: 808, top: 132, width: 388, height: 448 };
  addText(slide9, {
    text: "",
    left: 796,
    top: 120,
    width: 412,
    height: 472,
    fill: null,
    lineColor: COLORS.border,
    radius: "roundRect",
  });
  addTitle(slide9, {
    kicker: "Conclusion",
    title: "Final Conclusion",
    subtitle: "Closer runtime, real harmful outputs, and still no clean exact-Qwen replication.",
    dark: true,
  });
  addCard(slide9, {
    left: 56,
    top: 254,
    width: 220,
    height: 206,
    title: "Confirmed",
    body: "The later GPU/vLLM evaluation surfaced real harmful outputs and reduced the largest runtime gap.",
    accent: COLORS.cyan,
    dark: true,
  });
  addCard(slide9, {
    left: 300,
    top: 254,
    width: 220,
    height: 206,
    title: "Not confirmed",
    body: "The exact Qwen-family result stayed mixed, and some judged positives still needed manual verification.",
    accent: COLORS.orange,
    dark: true,
  });
  addCard(slide9, {
    left: 544,
    top: 254,
    width: 220,
    height: 206,
    title: "Interpretation",
    body: "Runtime, judge choice, prompt pathway, and evaluation method all materially affect the observed result.",
    accent: COLORS.green,
    dark: true,
  });
  addText(slide9, {
    text: "Conclusion: the GPU/vLLM follow-up surfaced harmful outputs, but the paper's Qwen-family FITD result was still not cleanly reproduced.",
    left: 56,
    top: 490,
    width: 708,
    height: 132,
    fontSize: 22,
    color: COLORS.bg,
    bold: true,
    fill: COLORS.cyan,
    radius: "roundRect",
  });
  slide9.speakerNotes.setText(speakerNotes.get(9));

  const slide10 = deck.slides.add();
  addSlideFrame(slide10, { number: 10, dark: false });
  addTitle(slide10, {
    kicker: "Next Steps",
    title: "Remaining Work",
    subtitle: "The most useful next steps are narrower and closer to the paper's original setup than simply broadening the matrix again.",
    dark: false,
  });
  addBulletList(slide10, {
    items: [
      "Manually audit every judged positive in the GPU/vLLM follow-up",
      "Run the exact Qwen-family models again on the author-chain path with the raw target preserved",
      "Move the judge and assistant defaults closer to the paper's original setup",
      "Replace the scaffold with the paper's full adaptive FITD loop if a closer runtime is available",
    ],
    left: 76,
    top: 240,
    width: 720,
    height: 250,
    fontSize: 24,
    color: COLORS.ink,
    dark: false,
  });
  addCard(slide10, {
    left: 838,
    top: 228,
    width: 356,
    height: 258,
    title: "Overall interpretation",
    body: "This is a stronger and more informative reproduction than the initial local result, but it remains partial rather than fully paper-faithful.",
    accent: COLORS.red,
    dark: false,
  });
  addText(slide10, {
    text: "Overall interpretation: closer reproduction, mixed Qwen evidence, and unresolved paper-faithfulness gaps.",
    left: 128,
    top: 550,
    width: 1020,
    height: 70,
    fontSize: 24,
    color: COLORS.bg,
    bold: true,
    fill: COLORS.orange,
    radius: "roundRect",
    align: "center",
    valign: "middle",
  });
  slide10.speakerNotes.setText(speakerNotes.get(10));

  const pptx = await PresentationFile.exportPptx(deck);
  await pptx.save(OUTPUT_PPTX);

  const imported = await PresentationFile.importPptx(await fs.readFile(OUTPUT_PPTX));
  if (imported.slides.count !== 10) {
    throw new Error(`Expected 10 slides, found ${imported.slides.count}`);
  }
}

buildDeck().catch((error) => {
  console.error(error);
  process.exit(1);
});
