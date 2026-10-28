export type PillTone = "accent" | "success" | "warning" | "danger" | "neutral";

export interface StatusMeta {
  label: string;
  tone: PillTone;
}

const STATUS_META: Record<string, StatusMeta> = {
  idle: { label: "待处理", tone: "neutral" },
  active: { label: "训练中", tone: "accent" },
  done: { label: "已完成", tone: "success" },
};

export function projectStatusMeta(status: string): StatusMeta {
  return STATUS_META[status] ?? { label: "待处理", tone: "neutral" };
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

export function formatUpdated(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const sameDay =
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate();

  if (sameDay) {
    return `今天 ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  const isYesterday =
    date.getFullYear() === yesterday.getFullYear() &&
    date.getMonth() === yesterday.getMonth() &&
    date.getDate() === yesterday.getDate();

  if (isYesterday) {
    return `昨天 ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  }

  return `${date.getMonth() + 1} 月 ${date.getDate()} 日`;
}
