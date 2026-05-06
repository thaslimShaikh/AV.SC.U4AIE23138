export type NotificationType = "Event" | "Result" | "Placement";

export interface Notification {
  id: string;
  type: NotificationType;
  message: string;
  timestamp: string;
  read: boolean;
}