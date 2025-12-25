import { hash, compare } from "bcryptjs";

// Utility functions for next auth

// function to hash the password
export async function hashPassword(password: string) {
  return hash(password, 12);
}

// fucntion to compare hashed password and plain text
export async function comparePasswords(
  plainPassword: string,
  hashedPassword: string,
) {
  return compare(plainPassword, hashedPassword);
}
