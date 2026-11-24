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

import { Component, ViewEncapsulation } from '@angular/core';

/** A wrapping row of filters and actions above a table or a list. */
@Component({
  selector: 'ui-toolbar',
  templateUrl: './toolbar.html',
  styleUrl: './toolbar.css',
  // The inputs and selects it sizes are projected by the page.
  encapsulation: ViewEncapsulation.None,
  host: { role: 'toolbar' },
})
export class UiToolbar {}
