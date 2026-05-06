import { Request, Response } from "express";

import { mockNotifications } from "../utils/mockNotifications";

import { getTopPriorityNotifications } from "../services/priorityService";

export const getNotifications = (
  req: Request,
  res: Response
): void => {
  res.status(200).json({
    success: true,
    count: mockNotifications.length,
    data: mockNotifications,
  });
};

export const getPriorityNotifications = (
  req: Request,
  res: Response
): void => {
  const topNotifications =
    getTopPriorityNotifications(mockNotifications);

  res.status(200).json({
    success: true,
    count: topNotifications.length,
    data: topNotifications,
  });
};