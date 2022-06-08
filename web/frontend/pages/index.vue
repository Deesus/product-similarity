<template>
    <v-row justify="center" align="center">
        <v-col cols="12" sm="8" md="6">
            <FileUpload />
        </v-col>
    </v-row>
</template>

<script>
import FileUpload from '~/components/FileUpload'

export default{
    name: 'IndexPage',
    components: { FileUpload }
}
</script>
