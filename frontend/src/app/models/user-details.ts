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

/** The shortest password the backoffice accepts (`MIN_PASSWORD_LENGTH` in `users.py`). */
export const MIN_PASSWORD_LENGTH = 8;

/**
 * A backoffice user as the API returns it. `has_password` is false for a user
 * who signs in through Google only; `/user/get_details` (the session) does not
 * carry it, `/users/all` and `/user/me` do.
 */
export interface UserDetails {
  id: string
  email: string
  name: string
  is_admin: boolean
  picture: string | null
  active_project_id?: string | null
  has_password?: boolean
  created_at?: string
  updated_at?: string
}

/**
 * What an administrator's user form sends to `/users/create` and
 * `/users/update/{id}`. The server hashes `password`; an empty one is left
 * out, so an edit keeps the stored password.
 */
export interface UserFields {
  email: string
  name: string
  picture: string | null
  is_admin: boolean
  password?: string
}

/** What a user may change on their own record through `PATCH /user/me`. */
export interface ProfileFields {
  name: string
  picture: string | null
}
