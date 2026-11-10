// Minimal Markdown -> DOCX converter with native Word (OMML) equations.
// Usage: node md2docx.js <in.md> <out.docx> <report|blueprint>
const fs = require("fs");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, LevelFormat, PageBreak, TableOfContents, Footer, Header,
  PageNumber, MathRun, MathFraction, MathRadical, MathSubScript, MathSuperScript,
  MathSubSuperScript, MathSum,
} = d;
const OMath = d.Math;

const [, , IN, OUT, MODE] = process.argv;
const src = fs.readFileSync(IN, "utf8").replace(/\r/g, "");
const lines = src.split("\n");

const FONT = "Calibri";
const MONO = "Consolas";
const ACCENT = "1F3864";
const CONTENT_W = 9026; // A4, 1" margins (DXA)

// ---------- inline formatting ----------
const ESC_STAR = "\u0000";
function inline(text, base = {}) {
  text = text.replace(/\\\*/g, ESC_STAR).replace(/\\\|/g, "|");
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)/g).filter((p) => p !== "");
  return parts.map((p) => {
    const fix = (s) => s.split(ESC_STAR).join("*");
    if (p.startsWith("**") && p.endsWith("**") && p.length > 4)
      return new TextRun({ ...base, text: fix(p.slice(2, -2)), bold: true });
    if (p.startsWith("`") && p.endsWith("`") && p.length > 2)
      return new TextRun({ ...base, text: fix(p.slice(1, -1)), font: MONO, size: base.size ? base.size - 2 : 19, color: "7A2E0E" });
    if (p.startsWith("*") && p.endsWith("*") && p.length > 2)
      return new TextRun({ ...base, text: fix(p.slice(1, -1)), italics: true });
    return new TextRun({ ...base, text: fix(p) });
  });
}

// ---------- math (mini-LaTeX: _{} ^{} \frac{}{} \sqrt{} \sum_{}^{}{}) ----------
const OPS = new Set([" ", "(", "[", "]", "{", "}", "=", "+", "−", "-", "·", ",", "/", "|", "<", ">", "≤", "≥", "∈", "∩", "⊙", "≈", "∼", ":", ";", "×"]);
function readGroup(s, i) {
  if (s[i] !== "{") return ["", i];
  let depth = 0, j = i;
  for (; j < s.length; j++) {
    if (s[j] === "{") depth++;
    else if (s[j] === "}") { depth--; if (depth === 0) break; }
  }
  return [s.slice(i + 1, j), j + 1];
}
function parseMath(s) {
  const out = [];
  let buf = "";
  const flush = () => { if (buf) { out.push(new MathRun(buf)); buf = ""; } };
  let i = 0;
  while (i < s.length) {
    if (s.startsWith("\\frac", i)) {
      flush();
      const [a, j] = readGroup(s, i + 5);
      const [b, k] = readGroup(s, j);
      out.push(new MathFraction({ numerator: parseMath(a), denominator: parseMath(b) }));
      i = k; continue;
    }
    if (s.startsWith("\\sqrt", i)) {
      flush();
      const [a, j] = readGroup(s, i + 5);
      out.push(new MathRadical({ children: parseMath(a) }));
      i = j; continue;
    }
    if (s.startsWith("\\sum", i)) {
      flush(); i += 4;
      let sub = null, sup = null, body = [];
      if (s[i] === "_") { const [a, j] = readGroup(s, i + 1); sub = a; i = j; }
      if (s[i] === "^") { const [a, j] = readGroup(s, i + 1); sup = a; i = j; }
      if (s[i] === "{") { const [a, j] = readGroup(s, i); body = parseMath(a); i = j; }
      const opts = { children: body.length ? body : [new MathRun(" ")] };
      if (sub) opts.subScript = parseMath(sub);
      if (sup) opts.superScript = parseMath(sup);
      out.push(new MathSum(opts));
      continue;
    }
    const c = s[i];
    if ((c === "_" || c === "^") && s[i + 1] === "{") {
      let k = buf.length;
      while (k > 0 && !OPS.has(buf[k - 1]) && buf[k - 1] !== ")") k--;
      let base = buf.slice(k);
      if (!base && buf.endsWith(")")) { base = ")"; k = buf.length - 1; }
      buf = buf.slice(0, k);
      flush();
      const [a, j] = readGroup(s, i + 1);
      i = j;
      let sub = null, sup = null;
      if (c === "_") sub = a; else sup = a;
      const other = c === "_" ? "^" : "_";
      if (s[i] === other && s[i + 1] === "{") {
        const [b, k2] = readGroup(s, i + 1);
        if (other === "^") sup = b; else sub = b;
        i = k2;
      }
      const baseC = [new MathRun(base || " ")];
      if (sub !== null && sup !== null) out.push(new MathSubSuperScript({ children: baseC, subScript: parseMath(sub), superScript: parseMath(sup) }));
      else if (sub !== null) out.push(new MathSubScript({ children: baseC, subScript: parseMath(sub) }));
      else out.push(new MathSuperScript({ children: baseC, superScript: parseMath(sup) }));
      continue;
    }
    buf += c; i++;
  }
  flush();
  return out;
}
function equation(tex) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 160 },
    children: [new OMath({ children: parseMath(tex.trim()) })],
  });
}

// ---------- tables ----------
function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
  return s.split(/(?<!\\)\|/).map((c) => c.trim());
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(rows) {
  const header = rows[0];
  const body = rows.slice(1);
  const ncol = header.length;
  const lens = Array(ncol).fill(6);
  rows.forEach((r) => r.forEach((c, i) => { if (i < ncol) lens[i] = Math.max(lens[i], Math.min(46, c.replace(/[`*]/g, "").length)); }));
  const weights = lens.map((l) => Math.max(7, l));
  const sum = weights.reduce((a, b) => a + b, 0);
  const widths = weights.map((w) => Math.floor((w / sum) * CONTENT_W));
  widths[ncol - 1] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  const size = ncol >= 8 ? 15 : ncol >= 6 ? 16 : 18;
  const mkCell = (txt, i, isHead) => new TableCell({
    borders,
    width: { size: widths[i], type: WidthType.DXA },
    shading: isHead ? { fill: "D9E2F3", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ spacing: { before: 0, after: 0 }, children: inline(txt, { size, bold: isHead || undefined, font: FONT }) })],
  });
  const trs = [new TableRow({ tableHeader: true, children: header.map((c, i) => mkCell(c, i, true)) })];
  body.forEach((r) => {
    const cells = [];
    for (let i = 0; i < ncol; i++) cells.push(mkCell(r[i] ?? "", i, false));
    trs.push(new TableRow({ children: cells }));
  });
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows: trs });
}

// ---------- title page ----------
function titlePage() {
  const P = (text, opts = {}) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: opts.before ?? 0, after: opts.after ?? 120 }, children: [new TextRun({ text, font: FONT, size: opts.size ?? 24, bold: opts.bold, color: opts.color, italics: opts.italics })] });
  const team = [
    ["Name", "Roll No.", "Role"],
    ["⟨Member 1⟩", "⟨ ⟩", "Data & feature engineering"],
    ["⟨Member 2⟩", "⟨ ⟩", "Modelling & evaluation"],
    ["⟨Member 3⟩", "⟨ ⟩", "Backend & real-time systems"],
    ["⟨Member 4⟩", "⟨ ⟩", "Frontend, XAI & MCP"],
  ];
  return [
    P("B.Tech Credit-Based Machine Learning Course · Project Bank #41", { before: 1600, size: 22, color: "595959" }),
    P("P.O.W.E.R.", { before: 600, size: 72, bold: true, color: ACCENT }),
    P("Predictive Optimization of Wattage for EV Resources", { size: 32, color: ACCENT, after: 360 }),
    P("An End-to-End Real-Time Machine Learning Engine for Electric Vehicle Charging Demand Forecasting and Grid Load Management", { size: 26, italics: true, after: 900 }),
    P("Technical Project Report", { size: 28, bold: true, after: 400 }),
    table(team),
    P("Project Guide: ⟨Name, Designation⟩", { before: 400 }),
    P("Department of ⟨Department⟩, ⟨Institute⟩"),
    P("Academic Year ⟨2026–27⟩"),
    P("Repository: github.com/aryanpol73/P.O.W.E.R", { size: 20, color: "595959", before: 200 }),
    new Paragraph({ children: [new PageBreak()] }),
  ];
}

// ---------- main parse ----------
const children = [];
let numInstance = 0;
let firstH1 = true;
let i = 0;
let currentListNumbered = null;
while (i < lines.length) {
  const line = lines[i];
  const t = line.trim();

  if (t === "[[TITLE]]") { children.push(...titlePage()); i++; continue; }
  if (t === "[[TOC]]") {
    children.push(new Paragraph({ pageBreakBefore: true, spacing: { before: 240, after: 160 }, children: [new TextRun({ text: "Table of Contents", font: FONT, size: 32, bold: true, color: ACCENT })] }));
    children.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }));
    i++; continue;
  }
  if (t === "" || t === "---") { currentListNumbered = null; i++; continue; }

  // fenced code
  if (t.startsWith("```")) {
    const lang = t.slice(3).trim();
    const buf = [];
    i++;
    while (i < lines.length && !lines[i].trim().startsWith("```")) { buf.push(lines[i]); i++; }
    i++;
    if (lang) children.push(new Paragraph({ spacing: { before: 120, after: 0 }, children: [new TextRun({ text: lang === "mermaid" ? "Mermaid diagram source (render with any Mermaid viewer)" : lang.toUpperCase(), font: FONT, size: 16, color: "808080", italics: true })] }));
    buf.forEach((b, k) => children.push(new Paragraph({
      spacing: { before: 0, after: 0, line: 240 },
      shading: { fill: "F2F2F2", type: ShadingType.CLEAR, color: "auto" },
      children: [new TextRun({ text: b.length ? b.replace(/\t/g, "    ") : " ", font: MONO, size: 15 })],
    })));
    children.push(new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }));
    continue;
  }

  // equation
  if (t.startsWith("$$") && t.endsWith("$$") && t.length > 4) { children.push(equation(t.slice(2, -2))); i++; continue; }

  // headings
  const hm = /^(#{1,4})\s+(.*)$/.exec(t);
  if (hm) {
    const level = hm[1].length;
    const text = hm[2];
    const heading = [HeadingLevel.HEADING_1, HeadingLevel.HEADING_2, HeadingLevel.HEADING_3, HeadingLevel.HEADING_4][level - 1];
    const pb = level === 1 && !(firstH1 && MODE !== "report");
    if (level === 1) firstH1 = false;
    children.push(new Paragraph({ heading, pageBreakBefore: pb, children: inline(text) }));
    i++; continue;
  }

  // table
  if (t.startsWith("|")) {
    const rows = [];
    while (i < lines.length && lines[i].trim().startsWith("|")) {
      const r = splitRow(lines[i]);
      if (!r.every((c) => /^:?-{2,}:?$/.test(c))) rows.push(r);
      i++;
    }
    children.push(table(rows));
    children.push(new Paragraph({ spacing: { before: 0, after: 120 }, children: [] }));
    continue;
  }

  // lists
  const bm = /^(\s*)[-•]\s+(.*)$/.exec(line);
  const nm = /^(\s*)(\d+)\.\s+(.*)$/.exec(line);
  if (bm) {
    const level = bm[1].length >= 2 ? 1 : 0;
    children.push(new Paragraph({ numbering: { reference: "bullets", level }, spacing: { before: 0, after: 60 }, children: inline(bm[2]) }));
    i++; continue;
  }
  if (nm) {
    if (currentListNumbered === null) { numInstance++; currentListNumbered = numInstance; }
    children.push(new Paragraph({ numbering: { reference: "numbers", level: 0, instance: currentListNumbered }, spacing: { before: 0, after: 60 }, children: inline(nm[3]) }));
    i++; continue;
  }

  // blockquote
  if (t.startsWith(">")) {
    children.push(new Paragraph({
      shading: { fill: "EAF1FB", type: ShadingType.CLEAR, color: "auto" },
      border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2E75B6", space: 6 } },
      spacing: { before: 120, after: 120 },
      children: inline(t.replace(/^>\s?/, "")),
    }));
    i++; continue;
  }

  // paragraph (merge consecutive text lines)
  const buf = [t];
  i++;
  while (i < lines.length) {
    const n = lines[i].trim();
    if (!n || n.startsWith("#") || n.startsWith("|") || n.startsWith("```") || n.startsWith("$$") || /^[-•]\s/.test(n) || /^\d+\.\s/.test(n) || n.startsWith(">") || n.startsWith("[[")) break;
    buf.push(n); i++;
  }
  children.push(new Paragraph({ spacing: { before: 0, after: 140 }, alignment: AlignmentType.JUSTIFIED, children: inline(buf.join(" ")) }));
}

const headerText = MODE === "report" ? "P.O.W.E.R. · Project #41 Technical Report" : "P.O.W.E.R. · Design Audit & System Blueprint";
const doc = new Document({
  creator: "P.O.W.E.R. team",
  title: MODE === "report" ? "P.O.W.E.R. Technical Report" : "P.O.W.E.R. Design Blueprint",
  features: { updateFields: process.env.UPDATE_FIELDS === "1" },
  styles: {
    default: { document: { run: { font: FONT, size: 21 }, paragraph: { spacing: { line: 276 } } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 32, bold: true, color: ACCENT }, paragraph: { spacing: { before: 240, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 26, bold: true, color: "2E75B6" }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 23, bold: true, color: "404040" }, paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
      { id: "Heading4", name: "Heading 4", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 21, bold: true, italics: true, color: "404040" }, paragraph: { spacing: { before: 120, after: 60 }, keepNext: true, outlineLevel: 3 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } },
      ] },
      { reference: "numbers", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 340 } } } },
      ] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: headerText, size: 16, color: "808080", font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], size: 16, color: "808080", font: FONT })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((b) => { fs.writeFileSync(OUT, b); console.log("wrote", OUT, b.length, "bytes,", children.length, "blocks"); });
