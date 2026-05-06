import { Request, Response } from "express";
import { mockNotifications } from "../utils/mockNotifications";

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