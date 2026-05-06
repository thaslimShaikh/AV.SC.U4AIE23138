import { Request, Response } from "express";

export const registerUser = async (
  req: Request,
  res: Response
): Promise<void> => {
  try {
    const { name, email } = req.body;

    if (!name || !email) {
      res.status(400).json({
        success: false,
        message: "Name and email are required",
      });

      return;
    }

    const mockToken = `token_${Date.now()}`;

    res.status(201).json({
      success: true,
      message: "User registered successfully",
      data: {
        id: Date.now(),
        name,
        email,
        token: mockToken,
      },
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Internal server error",
    });
  }
};