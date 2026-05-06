import express from "express";
import cors from "cors";

import loggerMiddleware from "./middleware/loggerMiddleware";

import authRoutes from "./routes/authRoutes";
import notificationRoutes from "./routes/notificationRoutes";

const app = express();

app.use(cors());
app.use(express.json());

app.use(loggerMiddleware);

app.use("/api/auth", authRoutes);
app.use("/api/notifications", notificationRoutes);

app.get("/", (req, res) => {
  res.status(200).json({
    success: true,
    message: "Notification Backend Running",
  });
});

export default app;