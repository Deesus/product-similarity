<template>
    <v-app light>
        <v-main>
            <v-container>
                <Nuxt />
            </v-container>
        </v-main>
        <v-footer
            elevation="2"
            class="py-4"
            outlined
            app
        >
            <v-row>
                <v-col cols="12" sm="4" offset-sm="4" class="d-flex justify-center align-center">
                    <span class="text--secondary">Copyright &copy; {{ new Date().getFullYear() }} Deepankara Reddy</span>
                </v-col>
                <v-col cols="12" sm="4" class="d-flex justify-center justify-sm-end">
                    <a href="https://www.linkedin.com/in/deesus" target="_blank">
                        <v-btn plain>
                            <v-img class="footer-icon cursor-pointer" src="/linkedin.svg" contain height="28" width="28" />
                        </v-btn>
                    </a>
                    <a href="https://github.com/deesus" target="_blank">
                        <v-btn plain>
                            <v-img class="footer-icon cursor-pointer" src="/github.svg" contain height="30" width="30" />
                        </v-btn>
                    </a>
                </v-col>
            </v-row>
        </v-footer>
    </v-app>
</template>

<script>
export default {
    name: 'DefaultLayout'
}
</script>

<style lang="scss">
    .container {
        height: 100%;
    }

    .footer-icon {
        opacity: 0.6;
        transition: opacity 0.3s ease-in-out;

        &:hover {
            opacity: 1;
        }
    }
</style>
