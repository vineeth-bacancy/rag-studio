/**
 * Display names for documents. The title comes from the PDF's own metadata
 * (served by the API); the doc_id is the fallback, never an invented name.
 */
import type { DocumentFile } from "@/lib/api";

export function docTitle(docId: string, docs: DocumentFile[] | undefined): string {
  return docs?.find((d) => d.doc_id === docId)?.title ?? docId;
}

export function pageImageUrl(docId: string, page: number, dpi = 110): string {
  return `/api/documents/${encodeURIComponent(docId)}/pages/${page}/image?dpi=${dpi}`;
}
