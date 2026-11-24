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

/**
 * The console's component library. Import what a page needs from here:
 *
 *   import { UiButton, UiCard, UiPageHeader } from '../../ui';
 *
 * Each component keeps its template and stylesheet in its own folder; ui.css
 * (imported by styles.css) carries only what no component can own. See the
 * README next to this file.
 */
export { UiAlert, type AlertTone } from './alert/alert';
export { UiAvatar, initialsOf } from './avatar/avatar';
export { UiBadge, UiStatusBadge, type BadgeTone } from './badge/badge';
export { UiButton, type ButtonSize, type ButtonVariant } from './button/button';
export { UiCard, type CardTone } from './card/card';
export { UiChip } from './chip/chip';
export { UiCopyButton } from './copy-button/copy-button';
export { UiDialog } from './dialog/dialog';
export { UiEmptyState } from './empty-state/empty-state';
export { UiField } from './field/field';
export { UiInfo } from './info/info';
export { UiLoading, UiSpinner } from './spinner/spinner';
export { UiMenu, UiMenuItem } from './menu/menu';
export { UiNavItem } from './nav-item/nav-item';
export { UiPageHeader } from './page-header/page-header';
export { UiProgress } from './progress/progress';
export { UiStat, UiStats } from './stat/stat';
export { UiPager, UiTable } from './table/table';
export { UiToolbar } from './toolbar/toolbar';
export { BarChart, fillDays, type ChartBar, type ChartSeries } from './bar-chart/bar-chart';
