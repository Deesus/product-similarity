<template>
    <div>
        <!-- ----- Header: ----- -->
        <v-row justify="center" align="center" class="py-6">
            <v-col cols="12" class="d-flex justify-center align-center">
                <h1 class="font-weight-light">
                    Product <span class="text--secondary">Image Search</span>
                </h1>
            </v-col>
        </v-row>

        <!-- ----- Upload image dropzone: ----- -->
        <v-row justify="center" align="center">
            <v-col cols="12" sm="8" md="6">
                <FileUpload @get-products="getProducts" />
            </v-col>
        </v-row>

        <!-- ----- Display results: ----- -->
        <v-row v-if="images.length" class="mt-15">
            <v-col cols="12">
                <ListResults :images="images" />
            </v-col>
        </v-row>
    </div>
</template>

<script>
import FileUpload from '~/components/FileUpload'
import ListResults from '~/components/ListResults'

export default {
    name: 'IndexPage',
    components: { FileUpload, ListResults },
    data() {
        return {
            images: []
        }
    },
    methods: {
        getProducts({ 'file-paths': filePaths }) {
            this.images = filePaths
        }
    }
}
</script>
