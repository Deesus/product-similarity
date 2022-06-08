module.exports = {
    root: true,
    env: {
        browser: true,
        node: true
    },
    parserOptions: {
        parser: '@babel/eslint-parser',
        requireConfigFile: false
    },
    extends: [
        '@nuxtjs',
        'plugin:nuxt/recommended'
    ],
    plugins: [],
    // custom rules:
    rules: {
        indent: ['warn', 4],
        'vue/html-indent': ['warn', 4],
        'keyword-spacing': ['off'],
        'space-before-function-paren': ['error', 'never'],
        'no-console': ['off']
    }
}
