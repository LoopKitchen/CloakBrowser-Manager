import { Download, FileUp, Loader2, Trash2, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import type { ProfileFile } from "../lib/api";

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 ? 1 : 0)} ${units[unit]}`;
}

/** Files attached to one profile, shared by the toolbar panel and the drop target. */
export function useProfileFiles(profileId: string) {
  const [files, setFiles] = useState<ProfileFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setFiles(await api.listProfileFiles(profileId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to list files");
    }
  }, [profileId]);

  const upload = useCallback(
    async (file: File) => {
      setUploading(true);
      setError(null);
      try {
        const added = await api.uploadProfileFile(profileId, file);
        setFiles((prev) => [added, ...prev.filter((f) => f.id !== added.id)]);
        return added;
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploading(false);
      }
    },
    [profileId],
  );

  const remove = useCallback(
    async (fileId: string) => {
      try {
        await api.deleteProfileFile(profileId, fileId);
        setFiles((prev) => prev.filter((f) => f.id !== fileId));
      } catch (err) {
        setError(err instanceof Error ? err.message : "Delete failed");
      }
    },
    [profileId],
  );

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { files, uploading, error, refresh, upload, remove, clearError: () => setError(null) };
}

interface ProfileFilesButtonProps {
  profileId: string;
  profileName: string;
  files: ProfileFile[];
  uploading: boolean;
  error: string | null;
  onUpload: (file: File) => Promise<unknown>;
  onRemove: (fileId: string) => Promise<void>;
  onClearError: () => void;
}

export function ProfileFilesButton({
  profileId,
  profileName,
  files,
  uploading,
  error,
  onUpload,
  onRemove,
  onClearError,
}: ProfileFilesButtonProps) {
  const [open, setOpen] = useState(false);
  const panelRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (e: MouseEvent) => {
      if (!panelRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setOpen(false);
      triggerRef.current?.focus();
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const pick = async (list: FileList | null) => {
    const chosen = list?.[0];
    if (chosen) await onUpload(chosen);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div ref={panelRef} className="relative flex items-center">
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        onChange={(e) => pick(e.target.files)}
        data-testid="profile-file-input"
      />
      <button
        ref={triggerRef}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="dialog"
        aria-expanded={open}
        title="Files available to this profile"
        className={`relative p-1 ${open || files.length ? "text-accent" : "text-gray-500 hover:text-gray-300"}`}
      >
        {uploading ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" />
        ) : (
          <FileUp className="h-3.5 w-3.5" />
        )}
        {files.length > 0 && !uploading && (
          <span className="absolute -right-1 -top-1 rounded-full bg-accent px-1 text-[9px] font-medium leading-[14px] text-white">
            {files.length}
          </span>
        )}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Profile files"
          className="absolute right-0 top-full z-20 mt-1 w-80 rounded-md border border-border bg-surface-2 shadow-lg"
        >
          <div className="flex items-center justify-between border-b border-border px-3 py-2">
            <span className="text-xs font-medium text-gray-200">Files for this profile</span>
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="text-gray-500 hover:text-gray-300"
              aria-label="Close"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          {error && (
            <div className="flex items-start justify-between gap-2 border-b border-border bg-red-600/10 px-3 py-2">
              <span className="text-xs text-red-400">{error}</span>
              <button type="button" onClick={onClearError} className="text-red-400" aria-label="Dismiss error">
                <X className="h-3 w-3" />
              </button>
            </div>
          )}

          <div className="max-h-64 overflow-y-auto">
            {files.length === 0 ? (
              <p className="px-3 py-4 text-xs text-gray-500">
                No files yet. Upload one, or drop it straight onto the screen.
              </p>
            ) : (
              files.map((file) => (
                <div key={file.id} className="flex items-center gap-2 px-3 py-2 hover:bg-surface-3">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs text-gray-200" title={file.name}>
                      {file.name}
                    </div>
                    <div className="text-[11px] text-gray-500">{formatSize(file.size)}</div>
                  </div>
                  <a
                    href={api.profileFileUrl(profileId, file.id)}
                    className="p-1 text-gray-500 hover:text-gray-300"
                    title="Download to this computer"
                  >
                    <Download className="h-3.5 w-3.5" />
                  </a>
                  <button
                    type="button"
                    onClick={() => onRemove(file.id)}
                    className="p-1 text-gray-500 hover:text-red-400"
                    title="Remove from the profile"
                    aria-label={`Remove ${file.name}`}
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </div>
              ))
            )}
          </div>

          <div className="border-t border-border px-3 py-2">
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              disabled={uploading}
              className="btn-secondary flex w-full items-center justify-center gap-1.5"
            >
              <FileUp className="h-3.5 w-3.5" />
              <span>{uploading ? "Uploading..." : "Upload a file"}</span>
            </button>
            <p className="mt-2 text-[11px] leading-snug text-gray-500">
              In the page's file dialog, pick{" "}
              <span className="text-gray-400">Uploads — {profileName}</span> in the sidebar.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
