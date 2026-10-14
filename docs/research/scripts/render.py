import sys, pymupdf

src, out_png, out_txt = sys.argv[1], sys.argv[2], sys.argv[3]
doc = pymupdf.open(src)
print("pages:", doc.page_count)
p = doc[1]  # page 2
pix = p.get_pixmap(dpi=100)
pix.save(out_png)
print("png:", pix.width, "x", pix.height)
with open(out_txt, "w") as f:
    for i in range(min(doc.page_count, 12)):
        f.write("\n\n##### PAGE %d\n" % (i + 1))
        f.write(doc[i].get_text())
