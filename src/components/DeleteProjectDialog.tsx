import { useState } from "react";
import { LoaderCircle, Trash2, X } from "lucide-react";

export function DeleteProjectDialog({
  projectName,
  onClose,
  onConfirm,
}: {
  projectName: string;
  onClose: () => void;
  onConfirm: () => Promise<unknown>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleConfirm = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await onConfirm();
      onClose();
    } catch (err) {
      setError(String(err));
      setBusy(false);
    }
  };

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onClose();
      }}
    >
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-project-title"
      >
        <div className="modal__header">
          <h2 id="delete-project-title">删除项目</h2>
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
          <p className="modal-copy">
            将删除项目「{projectName}」及其全部图片、任务记录与渲染结果，此操作不可恢复。
          </p>
          {error && <div className="import-notice">{error}</div>}
        </div>
        <div className="modal__footer">
          <button className="btn" type="button" onClick={onClose} disabled={busy}>
            取消
          </button>
          <button
            className="btn btn--danger"
            type="button"
            onClick={() => void handleConfirm()}
            disabled={busy}
          >
            {busy ? <LoaderCircle className="spin" size={16} /> : <Trash2 size={16} />}
            删除
          </button>
        </div>
      </div>
    </div>
  );
}
