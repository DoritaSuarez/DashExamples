import dash
import dash_html_components as html


class CustomDash(dash.Dash):
    def interpolate_index(self, **kwargs):
        # Inspect the arguments by printing them
        print(kwargs)
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script src="assets/custom_script.js"></script>
</head>
<body>
  <div id="graph"></div>
</body>


                {app_entry}
                {config}
                {scripts}
                {renderer}
</html>
        '''.format(
            app_entry=kwargs['app_entry'],
            config=kwargs['config'],
            scripts=kwargs['scripts'],
            renderer=kwargs['renderer'])

app = CustomDash()

app.layout = html.Div('Simple Dash App')

if __name__ == '__main__':
    app.run_server(debug=True)