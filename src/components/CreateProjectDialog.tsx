import { useEffect, useState, type FormEvent } from "react";
import { FolderPlus, LoaderCircle, X } from "lucide-react";

export function CreateProjectDialog({
  open,
  onClose,
  onSubmit,
}: {
  open: boolean;
  onClose: () => void;
  onSubmit: (name: string) => Promise<unknown>;
}) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setName("");
      setBusy(false);
      setError(null);
    }
  }, [open]);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = name.trim();
    if (!trimmed || busy) return;
    setBusy(true);
    setError(null);
    try {
      await onSubmit(trimmed);
      onClose();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  };

  if (!open) return null;

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onClose();
      }}
    >
      <form className="modal" onSubmit={handleSubmit}>
        <div className="modal__header">
          <h2>新建项目</h2>
          <button
            className="icon-btn"
            type="button"
            onClick={onClose}
            disabled={busy}
            aria-label="关闭"
          >
            <X size={16} />
          </button>
        </div>
        <div className="modal__body">
          <label className="field-label" htmlFor="new-project-name">
            项目名称
          </label>
          <input
            id="new-project-name"
            className="input"
            type="text"
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="例如：客厅环拍"
            autoFocus
            disabled={busy}
          />
          {error && <div className="import-notice">{error}</div>}
        </div>
        <div className="modal__footer">
          <button className="btn" type="button" onClick={onClose} disabled={busy}>
            取消
          </button>
          <button
            className="btn btn--primary"
            type="submit"
            disabled={busy || !name.trim()}
          >
            {busy ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <FolderPlus size={16} />
            )}
            创建
          </button>
        </div>
      </form>
    </div>
  );
}
