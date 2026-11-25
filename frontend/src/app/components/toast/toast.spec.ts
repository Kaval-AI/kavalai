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
import { TestBed, fakeAsync, tick } from '@angular/core/testing';
import { Toast } from './toast';
import { ToastService } from '../../services/toast-service';

describe('Toast', () => {
  it('shows each toast by its tone until it is dismissed or times out', fakeAsync(() => {
    const fixture = TestBed.createComponent(Toast);
    const service = TestBed.inject(ToastService);
    const el: HTMLElement = fixture.nativeElement;

    service.success('Saved');
    service.error('Could not save', 0);
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="toast-success"]')!.textContent).toContain('Saved');
    expect(el.querySelector('[data-testid="toast-error"]')!.getAttribute('tone')).toBe('error');

    tick(2500);
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="toast-success"]')).toBeNull();
    expect(el.querySelector('[data-testid="toast-error"]')).not.toBeNull();

    el.querySelector<HTMLButtonElement>('[data-testid="toast-error"] .toast-close')!.click();
    fixture.detectChanges();
    expect(el.querySelector('[data-testid="toast-error"]')).toBeNull();
  }));

  it('lets a plain message through as information', fakeAsync(() => {
    const service = TestBed.inject(ToastService);
    service.show('Hello');
    expect(service.toasts()[0].type).toBe('info');
    tick(2500);
    expect(service.toasts()).toEqual([]);
  }));
});
