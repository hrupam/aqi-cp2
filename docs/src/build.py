"""Build docs/CODE_WALKTHROUGH.md and docs/CODE_WALKTHROUGH.html from the parts in docs/src/."""
import re
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter

SRC = Path(__file__).parent
OUT = SRC.parent
parts = sorted(SRC.glob("[0-9][0-9]_*.md"))
body = "\n".join(p.read_text(encoding="utf-8") for p in parts)

# ---------------------------------------------------------------- Markdown with a contents list
heads = re.findall(r"^(##) (\d+\..*)$", body, flags=re.M)


def anchor(t):
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


toc_md = "## Contents\n\n" + "\n".join(f"- [{t}](#{anchor(t)})" for _, t in heads) + "\n\n---\n"
first_rule = body.index("\n---\n")
md_text = body[:first_rule] + "\n\n" + toc_md + body[first_rule + 5:]
(OUT / "CODE_WALKTHROUGH.md").write_text(md_text, encoding="utf-8")

# ---------------------------------------------------------------- HTML
maths = []


def keep(m):
    maths.append(m.group(0))
    return f"@@MATH{len(maths) - 1}@@"


LIST_RE = re.compile(r"^(\s*)([*-]|\d+\.)\s")


def normalise_lists(text: str) -> str:
    """Python-Markdown needs a blank line before a list and 4-space nesting, while the source
    follows CommonMark (lists may interrupt paragraphs, 2-3 space nesting). Convert, and lift
    code fences out of list items (Python-Markdown can't render fences nested in lists)."""
    out, stack, fence_indent, prev = [], [], None, "blank"
    for line in text.split("\n"):
        stripped = line.lstrip(" ")
        n = len(line) - len(stripped)
        if fence_indent is not None:                      # inside a code fence
            if stripped.startswith("```"):
                out += ["```", ""]
                if fence_indent > 0:               # the fence was lifted out of a list:
                    stack = []                     # following items start a fresh list
                fence_indent, prev = None, "blank"
            else:
                out.append(line[min(n, fence_indent):])
            continue
        if stripped.startswith("```"):
            if out and out[-1].strip():
                out.append("")
            out.append(stripped)
            fence_indent = n
            continue
        if not stripped:
            out.append("")
            prev = "blank"
            continue
        if LIST_RE.match(line):
            while stack and n < stack[-1]:
                stack.pop()
            if not stack or n > stack[-1]:
                stack.append(n)
            level = len(stack) - 1
            if prev in ("text", "cont") and not LIST_RE.match(out[-1] if out else ""):
                out.append("")
            out.append(" " * (4 * level) + stripped)
            prev = "list"
        elif n == 0:
            stack = []
            out.append(stripped)
            prev = "text"
        else:                                             # continuation inside a list item
            lvl = sum(1 for s in stack if s < n)
            if stripped.startswith("|") and prev in ("list",):
                out.append("")
            out.append(" " * (4 * max(lvl, 1)) + stripped)
            prev = "cont"
    return "\n".join(out)


body = normalise_lists(body)
protected = re.sub(r"\$\$.+?\$\$", keep, body, flags=re.S)
protected = re.sub(r"(?<![\\$])\$(?!\s)[^$\n]+?(?<!\s)\$", keep, protected)

md = markdown.Markdown(extensions=["tables", "fenced_code", "codehilite", "toc", "sane_lists"],
                       extension_configs={"codehilite": {"guess_lang": False, "css_class": "hl"},
                                          "toc": {"slugify": lambda v, sep: anchor(v), "toc_depth": "2-3"}})
html_body = md.convert(protected)
html_body = re.sub(r"@@MATH(\d+)@@", lambda m: maths[int(m.group(1))], html_body)
toc = md.toc
pyg = HtmlFormatter(style="friendly").get_style_defs(".hl")

title_match = re.search(r"<h1[^>]*>(.*?)</h1>", html_body, flags=re.S)
title = re.sub("<[^>]+>", "", title_match.group(1)) if title_match else "Code walkthrough"

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Checkpoint 2 Walkthrough – Kolkata PM2.5</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:ital,wght@0,400;0,700;1,400&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"
  onload="renderMathInElement(document.querySelector('main'),{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}]}})"></script>
<style>
:root {{
  --ink:#1B2A41; --sky:#EEF3F8; --paper:#FFFFFF; --haze:#66788F; --line:#DCE3EC; --signal:#2F6FDE;
  --code-bg:#F5F8FB; --tint:#E6EEF8;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --ink:#E4EAF2; --sky:#0F1724; --paper:#162133; --haze:#9AA9BD;
    --line:#2A3850; --signal:#6E9BF0; --code-bg:#111B2B; --tint:#1C2A40; }}
}}
:root[data-theme="dark"] {{ --ink:#E4EAF2; --sky:#0F1724; --paper:#162133; --haze:#9AA9BD;
    --line:#2A3850; --signal:#6E9BF0; --code-bg:#111B2B; --tint:#1C2A40; }}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{ margin:0; background:var(--sky); color:var(--ink);
  font-family:'Atkinson Hyperlegible','Segoe UI',sans-serif; font-size:17px; line-height:1.65; }}
.layout {{ display:grid; grid-template-columns: 300px minmax(0,1fr); max-width:1320px; margin:0 auto; }}
nav {{ position:sticky; top:0; height:100vh; overflow-y:auto; padding:28px 18px 40px 24px;
  border-right:1px solid var(--line); font-size:.9rem; }}
nav .brand {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.15rem; line-height:1.2; margin-bottom:4px; }}
nav .sub {{ color:var(--haze); font-size:.82rem; margin-bottom:18px; }}
nav ul {{ list-style:none; padding-left:0; margin:0; }}
nav ul ul {{ padding-left:14px; margin:2px 0 8px; display:none; }}
nav li.open > ul {{ display:block; }}
nav a {{ color:var(--ink); text-decoration:none; display:block; padding:3px 8px; border-radius:6px; }}
nav ul ul a {{ color:var(--haze); font-size:.84rem; padding:2px 8px; }}
nav a:hover {{ background:var(--tint); }}
nav a.active {{ background:var(--tint); color:var(--signal); font-weight:700; }}
main {{ padding:40px 56px 120px; min-width:0; }}
main > * {{ max-width:860px; }}
main > table, main > div.hl, main > pre {{ max-width:100%; }}
h1 {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:clamp(2rem,3.6vw,2.9rem);
  line-height:1.08; letter-spacing:-.02em; margin:0 0 18px; }}
h2 {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:800; font-size:1.75rem; letter-spacing:-.01em;
  margin:64px 0 14px; padding-top:18px; border-top:3px solid var(--ink); scroll-margin-top:16px; }}
h3 {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:600; font-size:1.25rem; margin:36px 0 10px;
  scroll-margin-top:16px; }}
p, li {{ max-width:78ch; }}
a {{ color:var(--signal); }}
hr {{ display:none; }}
strong {{ font-weight:700; }}
code {{ font-family:'JetBrains Mono',Consolas,monospace; font-size:.86em; background:var(--code-bg);
  border:1px solid var(--line); border-radius:4px; padding:.05em .3em; }}
pre code, .hl code {{ border:0; background:none; padding:0; font-size:.84rem; }}
.hl, pre {{ background:var(--code-bg) !important; border:1px solid var(--line); border-radius:10px;
  padding:14px 16px; overflow-x:auto; line-height:1.45; }}
.hl pre {{ margin:0; padding:0; border:0; background:none !important; }}
table {{ border-collapse:collapse; width:100%; margin:14px 0 22px; font-size:.92rem; background:var(--paper);
  border:1px solid var(--line); border-radius:10px; overflow:hidden; display:block; overflow-x:auto; }}
th, td {{ text-align:left; vertical-align:top; padding:8px 12px; border-bottom:1px solid var(--line); }}
th {{ background:var(--tint); font-weight:700; white-space:nowrap; }}
tr:last-child td {{ border-bottom:0; }}
blockquote {{ margin:18px 0; padding:10px 18px; background:var(--paper); border-left:4px solid var(--signal);
  border-radius:0 8px 8px 0; }}
blockquote p {{ margin:6px 0; }}
.katex-display {{ overflow-x:auto; overflow-y:hidden; padding:6px 0; }}
.toolbar {{ position:fixed; right:18px; bottom:18px; display:flex; gap:8px; }}
.toolbar button {{ font:inherit; font-size:.85rem; padding:7px 12px; border-radius:999px; border:1px solid var(--line);
  background:var(--paper); color:var(--ink); cursor:pointer; }}
.toolbar button:focus-visible, nav a:focus-visible {{ outline:2px solid var(--signal); outline-offset:2px; }}
{pyg}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) .hl * {{ color: inherit !important; }} }}
:root[data-theme="dark"] .hl * {{ color: inherit !important; }}
@media (max-width: 900px) {{
  .layout {{ display:block; }}
  nav {{ position:static; height:auto; border-right:0; border-bottom:1px solid var(--line); padding:18px 16px; }}
  nav ul ul {{ display:none !important; }}
  main {{ padding:24px 16px 80px; }}
  body {{ font-size:16px; }}
}}
@media print {{
  nav, .toolbar {{ display:none; }} .layout {{ display:block; }} main {{ padding:0; }}
  h2 {{ break-before:page; }} body {{ background:#fff; color:#000; font-size:11pt; }}
}}
</style></head>
<body>
<div class="layout">
<nav aria-label="Contents">
  <div class="brand">Kolkata Air Quality Early Warning</div>
  <div class="sub">Checkpoint 2: project &amp; code walkthrough</div>
  {toc}
</nav>
<main>
{html_body}
</main>
</div>
<div class="toolbar"><button id="theme" type="button">Dark mode</button><button type="button" onclick="window.print()">Print / PDF</button></div>
<script>
// highlight the current section in the contents and open its sub-list
const links = [...document.querySelectorAll('nav a')];
const byId = Object.fromEntries(links.map(a => [decodeURIComponent(a.hash.slice(1)), a]));
const heads = [...document.querySelectorAll('main h2[id], main h3[id]')];
function mark() {{
  let cur = heads[0];
  for (const h of heads) {{ if (h.getBoundingClientRect().top < 120) cur = h; }}
  links.forEach(a => a.classList.remove('active'));
  document.querySelectorAll('nav li.open').forEach(li => li.classList.remove('open'));
  const a = cur && byId[cur.id];
  if (a) {{ a.classList.add('active'); let li = a.closest('li');
    while (li) {{ li.classList.add('open'); li = li.parentElement.closest('li'); }} }}
}}
document.addEventListener('scroll', mark, {{passive:true}}); mark();
const btn = document.getElementById('theme');
function setTheme(t) {{ document.documentElement.dataset.theme = t; btn.textContent = t === 'dark' ? 'Light mode' : 'Dark mode';
  try {{ localStorage.setItem('walkthrough-theme', t); }} catch(e) {{}} }}
let saved = null; try {{ saved = localStorage.getItem('walkthrough-theme'); }} catch(e) {{}}
setTheme(saved || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
btn.onclick = () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
</script>
</body></html>"""
(OUT / "CODE_WALKTHROUGH.html").write_text(page, encoding="utf-8")
print("built", len(md_text.splitlines()), "md lines;", len(page) // 1024, "KB html;", len(maths), "formulas")
