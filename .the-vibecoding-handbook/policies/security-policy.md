# Security Policy

## Authentication
- All API routes require valid JWT
- Session timeout: 24 hours

## Data Protection
- Input validation on ALL user inputs
- Output encoding to prevent XSS
- Parameterized queries only

## Environment Variables
- .env files NEVER committed
