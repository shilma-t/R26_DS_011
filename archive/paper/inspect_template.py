import re
import zipfile
from collections import Counter

TEMPLATE = r"C:\Users\shafr\Desktop\conference-template-a4.docx"

z = zipfile.ZipFile(TEMPLATE)
print("PARTS:", [n for n in z.namelist() if n.startswith("word/")])

styles = z.read("word/styles.xml").decode("utf-8")
ids = sorted(set(re.findall(r'w:styleId="([^"]+)"', styles)))
print("\nSTYLE IDS:")
for i in ids:
    print("  ", i)

doc = z.read("word/document.xml").decode("utf-8")
used = Counter(re.findall(r'w:pStyle w:val="([^"]+)"', doc))
print("\nSTYLES USED IN BODY:")
for k, v in used.most_common():
    print("  %-28s %d" % (k, v))

m = re.search(r"<w:sectPr.*?</w:sectPr>", doc, re.S)
print("\nSECTPR:")
print(m.group(0) if m else "none")
