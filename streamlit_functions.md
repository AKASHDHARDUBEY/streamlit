# Streamlit Functions Reference

## Page Configuration

| Function | Description |
|---|---|
| `st.set_page_config()` | Sets page title, icon, layout (`wide`/`centered`), and sidebar state. Must be the first Streamlit call. |

---

## Text & Display

| Function | Description |
|---|---|
| `st.title()` | Displays a large H1-style title. |
| `st.header()` | Displays an H2-style header. |
| `st.subheader()` | Displays an H3-style subheader. |
| `st.text()` | Displays fixed-width plain text. |
| `st.markdown()` | Renders Markdown-formatted text (bold, italics, links, etc.). |
| `st.write()` | Smart display — renders text, DataFrames, dicts, charts, and more automatically. |
| `st.caption()` | Displays small caption/footnote text. |
| `st.code()` | Displays a code block with optional syntax highlighting. |
| `st.latex()` | Renders a LaTeX math expression. |
| `st.divider()` | Draws a horizontal rule to visually separate sections. |
| `st.echo()` | Displays both the code block and its output. |

---

## Input Widgets

| Function | Description |
|---|---|
| `st.text_input()` | Single-line text input field. |
| `st.text_area()` | Multi-line text input field. |
| `st.number_input()` | Numeric input with optional min, max, and step values. |
| `st.slider()` | Slider to select a numeric value or a range. |
| `st.select_slider()` | Slider that selects from a discrete list of values. |
| `st.checkbox()` | Boolean checkbox, returns `True`/`False`. |
| `st.radio()` | Radio buttons for selecting one option from a list. |
| `st.selectbox()` | Dropdown to select a single option. |
| `st.multiselect()` | Dropdown to select multiple options. |
| `st.button()` | Clickable button, returns `True` when clicked. |
| `st.download_button()` | Button that triggers a file download. |
| `st.file_uploader()` | Widget for uploading one or more files. |
| `st.camera_input()` | Captures a photo from the user's webcam. |
| `st.color_picker()` | Color picker widget, returns a hex color string. |
| `st.date_input()` | Calendar date picker. |
| `st.time_input()` | Time picker widget. |
| `st.toggle()` | A toggle switch (on/off), similar to checkbox. |
| `st.form()` | Groups widgets into a form; inputs are only submitted when the submit button is pressed. |
| `st.form_submit_button()` | Submit button used inside `st.form()`. |

---

## Data Display

| Function | Description |
|---|---|
| `st.dataframe()` | Displays an interactive, scrollable DataFrame. |
| `st.table()` | Displays a static, non-interactive table. |
| `st.metric()` | Shows a KPI metric with a value and optional delta indicator. |
| `st.json()` | Renders a JSON object in a collapsible tree view. |

---

## Charts & Maps

| Function | Description |
|---|---|
| `st.line_chart()` | Renders a line chart from a DataFrame or array. |
| `st.bar_chart()` | Renders a bar chart from a DataFrame or series. |
| `st.area_chart()` | Renders an area chart. |
| `st.scatter_chart()` | Renders a scatter plot. |
| `st.map()` | Plots points on an interactive map using lat/lon data. |
| `st.pyplot()` | Displays a Matplotlib figure. |
| `st.altair_chart()` | Renders an Altair chart object. |
| `st.plotly_chart()` | Renders a Plotly figure. |
| `st.bokeh_chart()` | Renders a Bokeh figure. |
| `st.pydeck_chart()` | Renders a PyDeck 3D map visualization. |
| `st.graphviz_chart()` | Renders a Graphviz diagram. |
| `st.vega_lite_chart()` | Renders a Vega-Lite chart from a spec dict. |

---

## Media

| Function | Description |
|---|---|
| `st.image()` | Displays one or more images (URL, file path, or numpy array). |
| `st.audio()` | Embeds an audio player. |
| `st.video()` | Embeds a video player. |
| `st.logo()` | Displays a logo in the sidebar header. |

---

## Layout & Containers

| Function | Description |
|---|---|
| `st.columns()` | Creates side-by-side columns; returns a list of column objects. |
| `st.tabs()` | Creates tabbed sections; returns a list of tab containers. |
| `st.expander()` | Collapsible section that hides/shows content. |
| `st.container()` | An invisible container to group elements or control render order. |
| `st.empty()` | A single-element placeholder that can be overwritten later. |
| `st.sidebar` | A persistent sidebar panel (use `st.sidebar.widget()`). |
| `st.popover()` | A button that opens a floating popover panel with content. |

---

## Status & Feedback

| Function | Description |
|---|---|
| `st.progress()` | Displays a progress bar (0–100). |
| `st.spinner()` | Shows a loading spinner while executing a block. |
| `st.success()` | Shows a green success message box. |
| `st.info()` | Shows a blue informational message box. |
| `st.warning()` | Shows a yellow warning message box. |
| `st.error()` | Shows a red error message box. |
| `st.exception()` | Displays a formatted exception/traceback. |
| `st.toast()` | Shows a brief popup notification in the corner. |
| `st.balloons()` | Launches a celebratory balloon animation. |
| `st.snow()` | Launches a snowfall animation. |

---

## State & Control Flow

| Function | Description |
|---|---|
| `st.session_state` | Dict-like object to persist data across reruns within a session. |
| `st.rerun()` | Immediately reruns the script from top to bottom. |
| `st.stop()` | Stops script execution at the current line. |
| `st.query_params` | Read/write URL query parameters. |
| `st.fragment()` | Decorator to mark a function as a partial-rerun fragment. |

---

## Caching & Performance

| Function | Description |
|---|---|
| `@st.cache_data` | Decorator that caches the return value of a function based on its arguments (for data). |
| `@st.cache_resource` | Decorator that caches a shared resource like a DB connection or ML model (created once per process). |

---

## Custom Components

| Function | Description |
|---|---|
| `st.components.v1.html()` | Renders a raw HTML string inside an iframe. |
| `st.components.v1.iframe()` | Embeds an external URL inside an iframe. |
| `st.components.v1.declare_component()` | Registers a custom frontend component. |

---

## Navigation & Pages (Multi-page Apps)

| Function | Description |
|---|---|
| `st.navigation()` | Defines a multi-page navigation structure programmatically. |
| `st.Page()` | Represents a single page in a multi-page app. |
| `st.page_link()` | Displays a clickable link to navigate to another page. |
| `st.switch_page()` | Programmatically navigates to another page. |

---

## Connections & Data

| Function | Description |
|---|---|
| `st.connection()` | Creates a managed connection to a data source (SQL, Snowflake, etc.). |

---

## Utilities

| Function | Description |
|---|---|
| `st.help()` | Displays the docstring for any Python object. |
| `st.context` | Provides request-level context such as headers and cookies. |
| `st.html()` | Renders an HTML string directly inline (without iframe). |
