import { Notification } from "../types/notification";

export const mockNotifications: Notification[] = [
  {
    id: "1",
    type: "Placement",
    message: "Microsoft hiring drive tomorrow",
    timestamp: new Date().toISOString(),
    read: false,
  },
  {
    id: "2",
    type: "Result",
    message: "Mid exam results published",
    timestamp: new Date().toISOString(),
    read: false,
  },
  {
    id: "3",
    type: "Event",
    message: "Tech Fest starts tomorrow",
    timestamp: new Date().toISOString(),
    read: true,
  },
];