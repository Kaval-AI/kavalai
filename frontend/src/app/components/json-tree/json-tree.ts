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
import { Component, Input, OnInit, QueryList, ViewChildren } from '@angular/core';
import { UiButton, UiCopyButton } from '../../ui';

/**
 * A collapsible view of a JSON value. The top level is open and carries
 * Copy, Expand all and Collapse all; every nested object or array starts
 * closed and opens on a click. Nested levels are instances of this component,
 * with `key` and `depth` set by their parent.
 */
@Component({
  selector: 'app-json-tree',
  standalone: true,
  imports: [UiButton, UiCopyButton],
  templateUrl: './json-tree.html',
  styleUrl: './json-tree.css',
})
export class JsonTreeComponent implements OnInit {
  private _data: any;
  @Input() set data(value: any) {
    this._data = value;
    this.processData();
  }
  get data(): any {
    return this._data;
  }
  @Input() key: string | number | null = null;
  @Input() isExpanded: boolean | null = null;
  @Input() depth: number = 0;

  @ViewChildren(JsonTreeComponent) childComponents!: QueryList<JsonTreeComponent>;

  isObject: boolean = false;
  isArray: boolean = false;
  isExpandable: boolean = false;
  formattedValue: string = '';
  type: string = '';
  private copiedText: string | null = null;

  children: { key: string | number; value: any }[] = [];

  ngOnInit(): void {
    if (this.isExpanded === null) {
      this.isExpanded = this.depth < 1;
    }
  }

  private processData(): void {
    this.isObject = false;
    this.isArray = false;
    this.isExpandable = false;
    this.children = [];

    this.type = typeof this.data;
    if (this.data === null || this.data === undefined) {
      this.type = 'null';
      this.formattedValue = 'null';
    } else if (Array.isArray(this.data)) {
      this.isArray = true;
      this.isObject = true;
      this.isExpandable = this.data.length > 0;
      this.formattedValue = this.data.length === 1 ? '[1 item]' : `[${this.data.length} items]`;
      this.children = this.data.map((value: any, index: number) => ({ key: index, value }));
    } else if (this.type === 'object') {
      this.isObject = true;
      const keys = Object.keys(this.data);
      this.isExpandable = keys.length > 0;
      this.formattedValue = keys.length === 1 ? '{1 key}' : `{${keys.length} keys}`;
      this.children = keys.map((key) => ({ key, value: this.data[key] }));
    } else if (this.type === 'string') {
      this.formattedValue = `"${this.data}"`;
    } else {
      this.formattedValue = String(this.data);
    }
    this.copiedText = null;
  }

  /** The value as indented JSON, what Copy puts on the clipboard; built once per value. */
  get jsonText(): string {
    this.copiedText ??= JSON.stringify(this.data ?? null, null, 2);
    return this.copiedText;
  }

  toggleExpand(event: Event): void {
    if (this.isExpandable) {
      this.isExpanded = !this.isExpanded;
    }
    event.stopPropagation();
  }

  expandAll(event: Event): void {
    this.setExpandedRecursive(true);
    event.stopPropagation();
  }

  collapseAll(event: Event): void {
    this.setExpandedRecursive(false);
    event.stopPropagation();
  }

  /**
   * Opening a level renders its children only on the next change detection,
   * so the children are reached a tick later.
   */
  private setExpandedRecursive(expanded: boolean): void {
    if (this.isExpandable) {
      this.isExpanded = expanded;
      setTimeout(() => {
        this.childComponents.forEach((child) => child.setExpandedRecursive(expanded));
      });
    }
  }
}
