import { Request, Response, NextFunction } from "express";

const loggerMiddleware = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const currentTime = new Date().toISOString();

  console.log("----------- REQUEST LOG -----------");
  console.log(`Time   : ${currentTime}`);
  console.log(`Method : ${req.method}`);
  console.log(`URL    : ${req.originalUrl}`);
  console.log(`IP     : ${req.ip}`);
  console.log("-----------------------------------");

  next();
};

export default loggerMiddleware;