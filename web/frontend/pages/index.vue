<template>
    <div>
        <!-- ----- Header: ----- -->
        <v-row justify="center" align="center" class="py-6">
            <v-col cols="12" class="d-flex justify-center align-center">
                <v-img class="img-package" src="/package.svg" />
                <h1 class="font-weight-light ml-5">
                    Product <span class="text--secondary">Image Search</span>
                </h1>
            </v-col>
        </v-row>

        <v-row justify="center" align="center">
            <v-col cols="12" sm="8" md="6">
                <FileUpload />
            </v-col>
        </v-row>
    </div>
</template>

<script>
import FileUpload from '~/components/FileUpload'

export default {
    name: 'IndexPage',
    components: { FileUpload }
}
</script>

<style scoped lang="scss">
    .img-package {
        width: 100px;
        max-width: 100px;
        height: auto;
    }
</style>
