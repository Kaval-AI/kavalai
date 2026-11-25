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
import { errorMessage } from './http-error';

function failure(status: number, error: unknown): HttpErrorResponse {
  return new HttpErrorResponse({ status, error });
}

describe('errorMessage', () => {
  it("reads FastAPI's detail", () => {
    expect(errorMessage(failure(400, { detail: 'Failed to connect: refused' }))).toBe(
      'Failed to connect: refused'
    );
  });

  it('names the field of a validation error', () => {
    const detail = [{ loc: ['body', 'email'], msg: 'Field required' }];
    expect(errorMessage(failure(422, { detail }))).toBe('email: Field required');
    expect(errorMessage(failure(422, { detail: [{ msg: 'Bad' }] }))).toBe('Bad');
  });

  it('says when the server cannot be reached', () => {
    expect(errorMessage(failure(0, null))).toBe('Cannot reach the server.');
  });

  it('falls back with the status code', () => {
    expect(errorMessage(failure(502, '<html>'), 'Could not load.')).toBe(
      'Could not load. (HTTP 502)'
    );
  });

  it('uses the message of a plain error, and the fallback otherwise', () => {
    expect(errorMessage(new Error('Boom'))).toBe('Boom');
    expect(errorMessage('nonsense', 'Fallback')).toBe('Fallback');
  });
});
