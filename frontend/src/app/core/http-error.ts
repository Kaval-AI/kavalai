/*
Copyright 2026 OÜ KAVAL AI (registry code 17393877)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
import { HttpErrorResponse } from '@angular/common/http';

/**
 * A short human-readable message for an API failure.
 *
 * FastAPI puts the reason in `detail`: a string for an `HTTPException`, a
 * list of field errors for a request that failed validation. Anything else
 * falls back to `fallback`, with the status code so two failures can be told
 * apart.
 */
export function errorMessage(
  error: unknown,
  fallback = 'Something went wrong. Please try again.'
): string {
  if (error instanceof HttpErrorResponse) {
    const detail = error.error?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && detail.length) {
      const first = detail[0];
      const field = Array.isArray(first?.loc)
        ? first.loc[first.loc.length - 1]
        : undefined;
      return field ? `${field}: ${first.msg}` : String(first?.msg ?? fallback);
    }
    if (error.status === 0) return 'Cannot reach the server.';
    return `${fallback} (HTTP ${error.status})`;
  }
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}
