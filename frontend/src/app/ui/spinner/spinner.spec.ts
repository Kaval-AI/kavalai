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

import { provideZonelessChangeDetection } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { UiLoading, UiSpinner } from './spinner';

describe('UiSpinner and UiLoading', () => {
  beforeEach(() => TestBed.configureTestingModule({ providers: [provideZonelessChangeDetection()] }));

  it('is a labelled status element, large inside ui-loading with its caption', () => {
    const spinner = TestBed.createComponent(UiSpinner);
    spinner.detectChanges();
    expect(spinner.nativeElement.getAttribute('role')).toBe('status');
    expect(Array.from((spinner.nativeElement as HTMLElement).classList)).toEqual(['ui-spinner']);
    const loading = TestBed.createComponent(UiLoading);
    loading.componentRef.setInput('label', 'Loading your chatbots');
    loading.detectChanges();
    expect(loading.nativeElement.querySelector('ui-spinner').className).toContain('ui-spinner-lg');
    expect(loading.nativeElement.textContent).toContain('Loading your chatbots');
  });
});
