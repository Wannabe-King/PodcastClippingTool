import { z } from "zod";
// Definintion of auth schema for zod resolve

export const signupSchema = z.object({
  email: z.string().email("Please enter a valid email address"),
  password: z.string().min(8, "Password must be at least 8 characters"),
});

export const loginSchema = z.object({
  email: z.string().email("Please enter a valid email address"),
  password: z.string().min(1, "Password is required"),
});

export type SignupFormValues = z.infer<typeof signupSchema>;
export type LoginFormValues = z.infer<typeof loginSchema>;
