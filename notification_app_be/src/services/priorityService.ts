import { Notification } from "../types/notification";

const typeWeights: Record<string, number> = {
  Placement: 3,
  Result: 2,
  Event: 1,
};

export const getPriorityScore = (
  notification: Notification
): number => {
  const typeScore = typeWeights[notification.type] || 0;

  const currentTime = new Date().getTime();

  const notificationTime = new Date(
    notification.timestamp
  ).getTime();

  const minutesDifference =
    (currentTime - notificationTime) / (1000 * 60);

  const recencyScore = Math.max(0, 100 - minutesDifference);

  return typeScore * 100 + recencyScore;
};

export const getTopPriorityNotifications = (
  notifications: Notification[],
  limit: number = 10
): Notification[] => {
  return notifications
    .sort((a, b) => {
      return (
        getPriorityScore(b) -
        getPriorityScore(a)
      );
    })
    .slice(0, limit);
};