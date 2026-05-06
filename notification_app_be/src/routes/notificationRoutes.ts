import express from "express";

import {
  getNotifications,
  getPriorityNotifications,
} from "../controllers/notificationController";

const router = express.Router();

router.get("/", getNotifications);

router.get(
  "/priority",
  getPriorityNotifications
);

export default router;